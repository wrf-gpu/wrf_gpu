"""v0.13 proof: RRTMG leading-column tiling is numerically inert.

This proof targets the remaining v0.13 1 km memory blocker after band/g-point
and taumol/optics chunking: the public SW/LW operational solves still accepted
the whole leading column batch at once, so LW transients scaled across every
nest column.  The implementation now flattens arbitrary leading dimensions,
scans over fixed-size column tiles, and reshapes outputs back to the caller's
layout.

Run modes::

    # CPU bit-inertness; writes proofs/v013/rrtmg_column_tile.json
    PYTHONPATH=src JAX_PLATFORMS=cpu python proofs/v013/rrtmg_column_tile.py --mode inertness

    # GPU peak-VRAM suite for the manager to run later
    XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
      python proofs/v013/rrtmg_column_tile.py --mode vram-suite --ncol 65536 --nrep 3 --tile-cols 16384
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs" / "v013" / "rrtmg_column_tile.json"

SW_FIELDS = (
    "heating_rate", "flux_down", "flux_up", "toa_down", "toa_up",
    "surface_down", "surface_up", "column_absorbed", "surface_absorbed",
    "surface_direct", "surface_diffuse", "surface_diffuse_fraction",
    "topographic_correction_factor", "surface_down_topographic",
    "surface_up_topographic", "surface_absorbed_topographic",
)
LW_FIELDS = (
    "heating_rate", "flux_down", "flux_up", "toa_down", "toa_up",
    "surface_down", "surface_up", "column_net_heating", "surface_emission",
)
SW_CLEAR_FIELDS = ("clear_flux_down", "clear_flux_up")
LW_CLEAR_FIELDS = ("clear_flux_down", "clear_flux_up")


def _enable_x64() -> None:
    from jax import config

    config.update("jax_enable_x64", True)


def _column_count(leading_shape: tuple[int, ...]) -> int:
    return int(np.prod(leading_shape, dtype=np.int64)) if leading_shape else 1


def _repeat_to_leading(arr: np.ndarray, leading_shape: tuple[int, ...], nrep: int = 1) -> np.ndarray:
    """Repeats the 3-column fixture into an arbitrary leading shape."""

    out = np.asarray(arr, dtype=np.float64)
    ncol = _column_count(leading_shape)
    if out.ndim >= 2:
        reps = (ncol + out.shape[0] - 1) // out.shape[0]
        out = np.tile(out, (reps,) + (1,) * (out.ndim - 1))[:ncol]
        if nrep > 1:
            out = np.repeat(out, nrep, axis=-1)
        return out.reshape(leading_shape + out.shape[1:])
    reps = (ncol + out.shape[0] - 1) // out.shape[0]
    out = np.tile(out, reps)[:ncol]
    return out.reshape(leading_shape)


def _sw_state(cls, arr: dict[str, np.ndarray], leading_shape: tuple[int, ...], nrep: int = 1):
    import jax.numpy as jnp

    def col(name: str):
        return jnp.asarray(_repeat_to_leading(arr[name], leading_shape, nrep))

    return cls(
        T=col("input_T"), p=col("input_p"), qv=col("input_qv"),
        qc=col("input_qc"), qi=col("input_qi"), qs=col("input_qs"),
        qg=col("input_qg"), cloud_fraction=col("input_cloud_fraction"),
        surface_albedo=col("input_surface_albedo"), coszen=col("input_coszen"),
        dz=col("input_dz"), rho=col("input_rho"),
    )


def _lw_state(cls, arr: dict[str, np.ndarray], leading_shape: tuple[int, ...], nrep: int = 1):
    import jax.numpy as jnp

    def col(name: str):
        return jnp.asarray(_repeat_to_leading(arr[name], leading_shape, nrep))

    return cls(
        T=col("input_T"), p=col("input_p"), qv=col("input_qv"),
        qc=col("input_qc"), qi=col("input_qi"), qs=col("input_qs"),
        qg=col("input_qg"), cloud_fraction=col("input_cloud_fraction"),
        surface_temperature=col("input_surface_temperature"),
        surface_emissivity=col("input_surface_emissivity"),
        dz=col("input_dz"), rho=col("input_rho"),
    )


def _sw_topography(cls, leading_shape: tuple[int, ...]):
    import jax.numpy as jnp

    ncol = _column_count(leading_shape)
    flat = np.linspace(0.0, 1.0, ncol, dtype=np.float64).reshape(leading_shape)
    return cls(
        latitude_deg=jnp.asarray(27.5 + 1.5 * flat),
        declination_rad=jnp.asarray(np.full(leading_shape, 0.21, dtype=np.float64)),
        hour_angle_rad=jnp.asarray(-0.35 + 0.7 * flat),
        slope_rad=jnp.asarray(0.02 + 0.08 * flat),
        slope_azimuth_rad=jnp.asarray(0.5 + 1.2 * flat),
        shadow_mask=jnp.asarray(np.ones(leading_shape, dtype=np.float64)),
    )


def _worst(candidate, reference, fields: tuple[str, ...]) -> dict[str, Any]:
    worst_abs = 0.0
    worst_rel = 0.0
    bit_equal = True
    per_field: dict[str, dict[str, Any]] = {}
    for field in fields:
        cand = np.asarray(getattr(candidate, field))
        ref = np.asarray(getattr(reference, field))
        diff = np.abs(cand - ref)
        rel = diff / np.maximum(np.abs(ref), 1.0e-30)
        max_abs = float(diff.max())
        max_rel = float(rel.max())
        equal = bool(np.array_equal(cand, ref))
        worst_abs = max(worst_abs, max_abs)
        worst_rel = max(worst_rel, max_rel)
        bit_equal = bit_equal and equal
        per_field[field] = {
            "shape": list(cand.shape),
            "dtype": str(cand.dtype),
            "max_abs": max_abs,
            "max_rel": max_rel,
            "bit_equal": equal,
        }
    return {
        "max_abs": worst_abs,
        "max_rel": worst_rel,
        "bit_equal": bool(bit_equal),
        "per_field": per_field,
    }


def _clear_cache(fn) -> None:
    clear = getattr(fn, "clear_cache", None)
    if clear is not None:
        clear()


def _set_sw_column_knobs(swmod, *, enabled: bool, tile_cols: int) -> None:
    swmod._SW_COLUMN_TILING = enabled
    swmod._SW_COLUMN_TILE_COLS = tile_cols
    _clear_cache(swmod.solve_rrtmg_sw_column)


def _set_lw_column_knobs(lwmod, *, enabled: bool, tile_cols: int) -> None:
    lwmod._LW_COLUMN_TILING = enabled
    lwmod._LW_COLUMN_TILE_COLS = tile_cols
    _clear_cache(lwmod.solve_rrtmg_lw_column)


def _tile_metadata(ncol: int, requested_tile_cols: int) -> dict[str, int]:
    tile_cols = min(max(int(requested_tile_cols), 1), ncol)
    n_tiles = (ncol + tile_cols - 1) // tile_cols
    padded_ncol = n_tiles * tile_cols
    return {
        "requested_tile_cols": int(requested_tile_cols),
        "effective_tile_cols": int(tile_cols),
        "n_tiles": int(n_tiles),
        "pad_cols": int(padded_ncol - ncol),
    }


def run_inertness() -> dict[str, Any]:
    _enable_x64()
    from gpuwrf.validation.tier1_rrtmg import _arrays, SW_SAMPLE, LW_SAMPLE
    import gpuwrf.physics.rrtmg_sw as swmod
    import gpuwrf.physics.rrtmg_lw as lwmod
    from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, RRTMGSWTopographyState
    from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState

    leading_shape = (2, 3)
    ncol = _column_count(leading_shape)
    sw_arr = _arrays(SW_SAMPLE)
    lw_arr = _arrays(LW_SAMPLE)
    sw_state = _sw_state(RRTMGSWColumnState, sw_arr, leading_shape)
    lw_state = _lw_state(RRTMGLWColumnState, lw_arr, leading_shape)
    topography = _sw_topography(RRTMGSWTopographyState, leading_shape)

    sw_orig = (swmod._SW_COLUMN_TILING, swmod._SW_COLUMN_TILE_COLS)
    lw_orig = (lwmod._LW_COLUMN_TILING, lwmod._LW_COLUMN_TILE_COLS)
    sw_default_tile = int(sw_orig[1])
    lw_default_tile = int(lw_orig[1])
    forced_tile = 4

    sw_rows = []
    lw_rows = []
    try:
        for with_clear_sky in (False, True):
            fields = SW_FIELDS + (SW_CLEAR_FIELDS if with_clear_sky else ())
            _set_sw_column_knobs(swmod, enabled=False, tile_cols=0)
            ref = swmod.solve_rrtmg_sw_column(sw_state, debug=False, with_clear_sky=with_clear_sky)
            for label, tile_cols in (("default", sw_default_tile), ("forced_tile4_pad", forced_tile)):
                _set_sw_column_knobs(swmod, enabled=True, tile_cols=tile_cols)
                cand = swmod.solve_rrtmg_sw_column(sw_state, debug=False, with_clear_sky=with_clear_sky)
                row = {
                    "case": "sw",
                    "candidate": label,
                    "with_clear_sky": bool(with_clear_sky),
                    "topography": False,
                    **_tile_metadata(ncol, tile_cols),
                    **_worst(cand, ref, fields),
                }
                sw_rows.append(row)

        _set_sw_column_knobs(swmod, enabled=False, tile_cols=0)
        topo_ref = swmod.solve_rrtmg_sw_column(sw_state, debug=False, topography=topography, with_clear_sky=True)
        _set_sw_column_knobs(swmod, enabled=True, tile_cols=forced_tile)
        topo_cand = swmod.solve_rrtmg_sw_column(sw_state, debug=False, topography=topography, with_clear_sky=True)
        sw_rows.append({
            "case": "sw",
            "candidate": "forced_tile4_pad_topography",
            "with_clear_sky": True,
            "topography": True,
            **_tile_metadata(ncol, forced_tile),
            **_worst(topo_cand, topo_ref, SW_FIELDS + SW_CLEAR_FIELDS),
        })

        for with_clear_sky in (False, True):
            fields = LW_FIELDS + (LW_CLEAR_FIELDS if with_clear_sky else ())
            _set_lw_column_knobs(lwmod, enabled=False, tile_cols=0)
            ref = lwmod.solve_rrtmg_lw_column(lw_state, debug=False, with_clear_sky=with_clear_sky)
            for label, tile_cols in (("default", lw_default_tile), ("forced_tile4_pad", forced_tile)):
                _set_lw_column_knobs(lwmod, enabled=True, tile_cols=tile_cols)
                cand = lwmod.solve_rrtmg_lw_column(lw_state, debug=False, with_clear_sky=with_clear_sky)
                row = {
                    "case": "lw",
                    "candidate": label,
                    "with_clear_sky": bool(with_clear_sky),
                    **_tile_metadata(ncol, tile_cols),
                    **_worst(cand, ref, fields),
                }
                lw_rows.append(row)
    finally:
        _set_sw_column_knobs(swmod, enabled=bool(sw_orig[0]), tile_cols=int(sw_orig[1]))
        _set_lw_column_knobs(lwmod, enabled=bool(lw_orig[0]), tile_cols=int(lw_orig[1]))

    rows = sw_rows + lw_rows
    return {
        "proof": "v0.13 RRTMG leading-column tiling CPU inertness",
        "branch": "worker/gpt/v013-rrtmg-coltile",
        "mode": "inertness",
        "platform": "cpu",
        "leading_shape": list(leading_shape),
        "ncol": ncol,
        "reference": "public solver with column tiling disabled (_*_COLUMN_TILING=False, _*_COLUMN_TILE_COLS=0)",
        "candidates": {
            "default": "module default column tiling enabled; one effective tile on this small fixture",
            "forced_tile4_pad": "column tiling enabled with tile_cols=4; two scan tiles and two padded columns",
        },
        "defaults": {
            "sw_column_tile_cols": sw_default_tile,
            "lw_column_tile_cols": lw_default_tile,
        },
        "sw": {
            "rows": sw_rows,
            "all_bit_identical": all(r["bit_equal"] for r in sw_rows),
            "max_abs_over_all": max(r["max_abs"] for r in sw_rows),
            "max_rel_over_all": max(r["max_rel"] for r in sw_rows),
        },
        "lw": {
            "rows": lw_rows,
            "all_bit_identical": all(r["bit_equal"] for r in lw_rows),
            "max_abs_over_all": max(r["max_abs"] for r in lw_rows),
            "max_rel_over_all": max(r["max_rel"] for r in lw_rows),
        },
        "verdict": {
            "sw_all_sky_bit_identical": all(r["bit_equal"] for r in sw_rows if not r["with_clear_sky"] and not r.get("topography")),
            "sw_clear_sky_bit_identical": all(r["bit_equal"] for r in sw_rows if r["with_clear_sky"] and not r.get("topography")),
            "lw_all_sky_bit_identical": all(r["bit_equal"] for r in lw_rows if not r["with_clear_sky"]),
            "lw_clear_sky_bit_identical": all(r["bit_equal"] for r in lw_rows if r["with_clear_sky"]),
            "all_required_cases_bit_identical": all(r["bit_equal"] for r in rows),
            "max_abs_over_all": max(r["max_abs"] for r in rows),
            "max_rel_over_all": max(r["max_rel"] for r in rows),
        },
        "gpu_proof_command": (
            "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src "
            "python proofs/v013/rrtmg_column_tile.py --mode vram-suite "
            "--ncol 65536 --nrep 3 --tile-cols 16384"
        ),
    }


def run_vram(kind: str, column_mode: str, ncol: int, nrep: int, tile_cols: int) -> dict[str, Any]:
    """Measures one GPU peak-memory config in a fresh process."""

    _enable_x64()
    import jax

    enabled = column_mode == "tiled"
    if kind == "sw":
        import gpuwrf.physics.rrtmg_sw as swmod
        from gpuwrf.validation.tier1_rrtmg import _arrays, SW_SAMPLE
        from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState

        _set_sw_column_knobs(swmod, enabled=enabled, tile_cols=tile_cols if enabled else 0)
        state = _sw_state(RRTMGSWColumnState, _arrays(SW_SAMPLE), (ncol,), nrep=nrep)
        fields = SW_FIELDS
        solve = lambda s: swmod.solve_rrtmg_sw_column(s, debug=False)
    else:
        import gpuwrf.physics.rrtmg_lw as lwmod
        from gpuwrf.validation.tier1_rrtmg import _arrays, LW_SAMPLE
        from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState

        _set_lw_column_knobs(lwmod, enabled=enabled, tile_cols=tile_cols if enabled else 0)
        state = _lw_state(RRTMGLWColumnState, _arrays(LW_SAMPLE), (ncol,), nrep=nrep)
        fields = LW_FIELDS
        solve = lambda s: lwmod.solve_rrtmg_lw_column(s, debug=False)

    dev = jax.devices()[0]
    try:
        dev.memory_stats()
    except Exception:
        pass
    out = solve(state)
    for field in fields:
        jax.block_until_ready(getattr(out, field))
    stats = dev.memory_stats() or {}
    return {
        "kind": kind,
        "column_mode": column_mode,
        "ncol": int(ncol),
        "nlev": int(np.asarray(state.p).shape[-1]),
        "tile_cols": int(tile_cols if enabled else 0),
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
        "bytes_in_use_after": int(stats.get("bytes_in_use", 0)),
        "peak_mib": round(int(stats.get("peak_bytes_in_use", 0)) / (1024 * 1024), 2),
    }


def run_vram_suite(ncol: int, nrep: int, tile_cols: int) -> dict[str, Any]:
    """Runs SW/LW tiled-vs-untiled VRAM configs in fresh subprocesses."""

    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("PYTHONPATH", "src")
    rows = []
    for kind in ("lw", "sw"):
        for column_mode in ("untiled", "tiled"):
            proc = subprocess.run(
                [
                    sys.executable, str(Path(__file__).resolve()),
                    "--mode", "vram",
                    "--kind", kind,
                    "--column-mode", column_mode,
                    "--ncol", str(ncol),
                    "--nrep", str(nrep),
                    "--tile-cols", str(tile_cols),
                    "--emit-json",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                env=env,
            )
            line = ""
            for stdout_line in proc.stdout.splitlines():
                if stdout_line.startswith("VRAM_JSON:"):
                    line = stdout_line[len("VRAM_JSON:"):]
            if line:
                rows.append(json.loads(line))
            else:
                oom = any("RESOURCE_EXHAUSTED" in s or "Out of memory" in s for s in proc.stderr.splitlines())
                rows.append({
                    "kind": kind,
                    "column_mode": column_mode,
                    "ncol": ncol,
                    "nrep": nrep,
                    "tile_cols": tile_cols if column_mode == "tiled" else 0,
                    "result": "OOM" if oom else "error",
                    "rc": proc.returncode,
                    "error_tail": proc.stderr.strip().splitlines()[-5:],
                })
    return {
        "proof": "v0.13 RRTMG leading-column tiling GPU peak-VRAM suite",
        "mode": "vram-suite",
        "ncol": ncol,
        "nrep": nrep,
        "tile_cols": tile_cols,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inertness", "vram", "vram-suite"), default="inertness")
    parser.add_argument("--kind", choices=("sw", "lw"), default="lw")
    parser.add_argument("--column-mode", choices=("tiled", "untiled"), default="tiled")
    parser.add_argument("--ncol", type=int, default=65536)
    parser.add_argument("--nrep", type=int, default=3)
    parser.add_argument("--tile-cols", type=int, default=16384)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()

    if args.mode == "inertness":
        record = run_inertness()
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
        print(f"\nwrote {OUT_JSON}")
    elif args.mode == "vram":
        record = run_vram(args.kind, args.column_mode, args.ncol, args.nrep, args.tile_cols)
        if args.emit_json:
            print("VRAM_JSON:" + json.dumps(record))
        else:
            print(json.dumps(record, indent=2))
    else:
        record = run_vram_suite(args.ncol, args.nrep, args.tile_cols)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
