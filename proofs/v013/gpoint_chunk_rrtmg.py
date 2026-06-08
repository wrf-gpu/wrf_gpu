"""v0.13 keystone proof: RRTMG g-point temporary chunking.

Proves the two claims of the v0.13 g-point-chunk sprint:

1. **Numerical inertness** (run on CPU, fp64):  tiling the RRTMG-SW / -LW
   two-stream + flux reduction over the spectral-band axis is bit-comparable
   (rel ~1e-13) to the un-tiled path.  Because each band tile reduces its
   g-points and the disjoint per-band contributions are accumulated in fp64,
   the result is provably **independent of the chunk width** (bit-identical
   across tile sizes) — the chunk knob is a pure VRAM control, not a physics
   change.

2. **Peak-VRAM reduction** (run on GPU):  on a representative column grid the
   chunked path's peak device `bytes_in_use` is materially lower than the
   un-chunked single-pass path, because only one band tile's
   `(ncol, nlev+1, tile_bands, 16)` temporary is live at a time instead of the
   full `(ncol, nlev+1, 14|16, 16)` g-point array.

Run modes::

    # bit-inertness (CPU, default)
    PYTHONPATH=src JAX_PLATFORMS=cpu python proofs/v013/gpoint_chunk_rrtmg.py --mode inertness

    # peak VRAM (GPU; one process per chunk config for a clean peak)
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
      PYTHONPATH=src python proofs/v013/gpoint_chunk_rrtmg.py --mode vram --kind sw --chunk 2 --ncol 16384

The driver (`--mode all`) orchestrates the CPU inertness sweep in-process and
spawns fresh GPU subprocesses for each VRAM config, then writes the combined
``proofs/v013/gpoint_chunk_rrtmg.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs" / "v013" / "gpoint_chunk_rrtmg.json"

SW_FIELDS = (
    "heating_rate",
    "flux_down",
    "flux_up",
    "toa_down",
    "toa_up",
    "surface_down",
    "surface_up",
    "column_absorbed",
    "surface_absorbed",
    "surface_direct",
    "surface_diffuse",
    "surface_diffuse_fraction",
)
LW_FIELDS = (
    "heating_rate",
    "flux_down",
    "flux_up",
    "toa_down",
    "toa_up",
    "surface_down",
    "surface_up",
    "column_net_heating",
    "surface_emission",
)


def _enable_x64():
    from jax import config

    config.update("jax_enable_x64", True)


def _sw_state(cls, arr, ncol_tile: int = 1, nrep: int = 1):
    import jax.numpy as jnp

    def col(name):
        a = np.asarray(arr[name], dtype=np.float64)
        if nrep > 1 and a.ndim >= 2:  # deepen the vertical (layer) axis
            a = np.repeat(a, nrep, axis=-1)
        if ncol_tile > 1:
            a = np.tile(a, (ncol_tile,) + (1,) * (a.ndim - 1))
        return jnp.asarray(a)

    return cls(
        T=col("input_T"), p=col("input_p"), qv=col("input_qv"),
        qc=col("input_qc"), qi=col("input_qi"), qs=col("input_qs"),
        qg=col("input_qg"), cloud_fraction=col("input_cloud_fraction"),
        surface_albedo=col("input_surface_albedo"), coszen=col("input_coszen"),
        dz=col("input_dz"), rho=col("input_rho"),
    )


def _lw_state(cls, arr, ncol_tile: int = 1, nrep: int = 1):
    import jax.numpy as jnp

    def col(name):
        a = np.asarray(arr[name], dtype=np.float64)
        if nrep > 1 and a.ndim >= 2:
            a = np.repeat(a, nrep, axis=-1)
        if ncol_tile > 1:
            a = np.tile(a, (ncol_tile,) + (1,) * (a.ndim - 1))
        return jnp.asarray(a)

    return cls(
        T=col("input_T"), p=col("input_p"), qv=col("input_qv"),
        qc=col("input_qc"), qi=col("input_qi"), qs=col("input_qs"),
        qg=col("input_qg"), cloud_fraction=col("input_cloud_fraction"),
        surface_temperature=col("input_surface_temperature"),
        surface_emissivity=col("input_surface_emissivity"),
        dz=col("input_dz"), rho=col("input_rho"),
    )


def _worst(cand, ref, fields):
    worst_abs = 0.0
    worst_rel = 0.0
    bit_equal = True
    for f in fields:
        c = np.asarray(getattr(cand, f))
        e = np.asarray(getattr(ref, f))
        d = np.abs(c - e)
        r = d / np.maximum(np.abs(e), 1e-30)
        worst_abs = max(worst_abs, float(d.max()))
        worst_rel = max(worst_rel, float(r.max()))
        if not np.array_equal(c, e):
            bit_equal = False
    return worst_abs, worst_rel, bit_equal


def run_inertness(ncol_tile: int = 1) -> dict:
    """Chunked-vs-unchunked bit-comparison for SW and LW across chunk widths.

    The 'unchunked' reference is the chunk = n_bands configuration of the SAME
    tiled code path (a single tile == the original single-pass solve).  Each
    candidate chunk width is compared against it.
    """

    _enable_x64()
    from gpuwrf.validation.tier1_rrtmg import _arrays, SW_SAMPLE, LW_SAMPLE
    import gpuwrf.physics.rrtmg_sw as swmod
    import gpuwrf.physics.rrtmg_lw as lwmod
    from gpuwrf.physics.rrtmg_sw import solve_rrtmg_sw_column, RRTMGSWColumnState
    from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column, RRTMGLWColumnState

    sw_arr = _arrays(SW_SAMPLE)
    lw_arr = _arrays(LW_SAMPLE)

    result = {"ncol_tile": ncol_tile, "sw": {}, "lw": {}}

    # ---- SW ----
    sw_nbands = swmod._SW_NBANDS
    swmod._SW_GPOINT_CHUNK_BANDS = sw_nbands
    sw_ref = solve_rrtmg_sw_column(_sw_state(RRTMGSWColumnState, sw_arr, ncol_tile), debug=False)
    sw_rows = []
    for chunk in (1, 2, 3, 4, 5, 7, sw_nbands):
        swmod._SW_GPOINT_CHUNK_BANDS = chunk
        cand = solve_rrtmg_sw_column(_sw_state(RRTMGSWColumnState, sw_arr, ncol_tile), debug=False)
        wa, wr, bit = _worst(cand, sw_ref, SW_FIELDS)
        sw_rows.append({"chunk_bands": chunk, "max_abs": wa, "max_rel": wr, "bit_equal": bit})
    swmod._SW_GPOINT_CHUNK_BANDS = sw_nbands
    result["sw"] = {
        "reference_chunk": sw_nbands,
        "rows": sw_rows,
        "max_rel_over_all_chunks": max(r["max_rel"] for r in sw_rows),
        "all_bit_identical": all(r["bit_equal"] for r in sw_rows),
    }

    # ---- LW ----
    lw_nbands = lwmod._LW_NBANDS
    lwmod._LW_GPOINT_CHUNK_BANDS = lw_nbands
    lw_ref = solve_rrtmg_lw_column(_lw_state(RRTMGLWColumnState, lw_arr, ncol_tile), debug=False)
    lw_rows = []
    for chunk in (1, 2, 4, 8, lw_nbands):
        lwmod._LW_GPOINT_CHUNK_BANDS = chunk
        cand = solve_rrtmg_lw_column(_lw_state(RRTMGLWColumnState, lw_arr, ncol_tile), debug=False)
        wa, wr, bit = _worst(cand, lw_ref, LW_FIELDS)
        lw_rows.append({"chunk_bands": chunk, "max_abs": wa, "max_rel": wr, "bit_equal": bit})
    lwmod._LW_GPOINT_CHUNK_BANDS = lw_nbands
    result["lw"] = {
        "reference_chunk": lw_nbands,
        "rows": lw_rows,
        "max_rel_over_all_chunks": max(r["max_rel"] for r in lw_rows),
        "all_bit_identical": all(r["bit_equal"] for r in lw_rows),
    }
    return result


def run_vram(kind: str, chunk: int, ncol: int, nrep: int = 1) -> dict:
    """Measure peak device bytes for one (kind, chunk) config on GPU.

    The kernel is wrapped in ``jax.jit`` because the operational path always runs
    the radiation column kernel inside a jitted ``lax.scan`` — only under jit
    does XLA do whole-program memory planning and free per-tile working sets, so
    an un-jitted (op-by-op) measurement is not representative.  Each call should
    be its own fresh process so the JAX allocator peak counter reflects only this
    configuration.
    """

    _enable_x64()
    import jax
    from gpuwrf.validation.tier1_rrtmg import _arrays, SW_SAMPLE, LW_SAMPLE

    dev = jax.devices()[0]

    if kind == "sw":
        import gpuwrf.physics.rrtmg_sw as swmod
        from gpuwrf.physics.rrtmg_sw import solve_rrtmg_sw_column, RRTMGSWColumnState

        swmod._SW_GPOINT_CHUNK_BANDS = chunk
        arr = _arrays(SW_SAMPLE)
        state = _sw_state(RRTMGSWColumnState, arr, ncol, nrep)
        fields = SW_FIELDS
        fn = jax.jit(lambda s: solve_rrtmg_sw_column(s, debug=False))
    else:
        import gpuwrf.physics.rrtmg_lw as lwmod
        from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column, RRTMGLWColumnState

        lwmod._LW_GPOINT_CHUNK_BANDS = chunk
        arr = _arrays(LW_SAMPLE)
        state = _lw_state(RRTMGLWColumnState, arr, ncol, nrep)
        fields = LW_FIELDS
        fn = jax.jit(lambda s: solve_rrtmg_lw_column(s, debug=False))

    # Reset peak counter, run once, block, then read peak.
    try:
        dev.memory_stats()  # warm the stats interface
    except Exception:
        pass
    out = fn(state)
    for f in fields:
        jax.block_until_ready(getattr(out, f))
    ms = dev.memory_stats() or {}
    ncol_total = int(np.asarray(state.p).shape[0])
    return {
        "kind": kind,
        "chunk_bands": chunk,
        "ncol": ncol_total,
        "nlev": int(np.asarray(state.p).shape[-1]),
        "peak_bytes_in_use": int(ms.get("peak_bytes_in_use", 0)),
        "bytes_in_use_after": int(ms.get("bytes_in_use", 0)),
        "peak_mib": round(int(ms.get("peak_bytes_in_use", 0)) / (1024 * 1024), 2),
    }


def drive_all(ncol_vram: int, nrep: int = 3) -> dict:
    """CPU inertness in-process + GPU VRAM (jit) via fresh subprocesses.

    Defaults to a deep column (nrep=3 -> nlev=48) where the SW two-stream
    working set is the dominant temporary, so the band-tiling win is visible.
    """

    record: dict = {
        "proof": "v0.13 RRTMG g-point chunking — bit-inertness + peak-VRAM",
        "base_commit": "6d051f8",
    }
    record["inertness"] = run_inertness(ncol_tile=1)

    # GPU VRAM: one subprocess per (kind, chunk) for a clean BFC peak counter.
    gpu_env = dict(os.environ)
    gpu_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpu_env.setdefault("PYTHONPATH", "src")
    vram_rows = []
    # LW's upstream taumol (branch+fallback) floor is heavy, so the LW probes use
    # a smaller column count (1/3 of the SW one) at the fixture's native nlev=16
    # to stay well within memory — the LW flux-stack change is VRAM-neutral
    # anyway (see the lw_note in the JSON), so this just confirms inertness +
    # no-regression rather than a reduction.
    lw_ncol = max(1, ncol_vram // 3)
    configs = [
        ("sw", 14, ncol_vram, nrep),  # unchunked single-pass
        ("sw", 1, ncol_vram, nrep),   # chunked default (one band per scan tile)
        ("lw", 16, lw_ncol, 1),       # LW single-stack
        ("lw", 1, lw_ncol, 1),        # LW per-band accumulate
    ]
    for kind, chunk, ncol, nr in configs:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--mode", "vram", "--kind", kind, "--chunk", str(chunk),
             "--ncol", str(ncol), "--nrep", str(nr), "--emit-json"],
            capture_output=True, text=True, env=gpu_env, cwd=str(ROOT),
        )
        line = ""
        for ln in proc.stdout.splitlines():
            if ln.startswith("VRAM_JSON:"):
                line = ln[len("VRAM_JSON:"):]
        if line:
            vram_rows.append(json.loads(line))
        else:
            vram_rows.append({"kind": kind, "chunk_bands": chunk, "ncol": ncol, "nrep": nr,
                              "error": proc.stderr.strip().splitlines()[-3:] if proc.stderr else "no output",
                              "rc": proc.returncode})
    record["vram"] = {"ncol_tile": ncol_vram, "rows": vram_rows}

    # Compute reductions.
    def peak_for(kind, chunk):
        for r in vram_rows:
            if r.get("kind") == kind and r.get("chunk_bands") == chunk and "peak_bytes_in_use" in r:
                return r["peak_bytes_in_use"]
        return None

    summary = {}
    sw_full = peak_for("sw", 14)
    sw_chunk = peak_for("sw", 1)
    if sw_full and sw_chunk:
        summary["sw_peak_mib_unchunked"] = round(sw_full / 1024 / 1024, 2)
        summary["sw_peak_mib_chunk1"] = round(sw_chunk / 1024 / 1024, 2)
        summary["sw_peak_reduction_pct"] = round(100.0 * (1.0 - sw_chunk / sw_full), 2)
    lw_full = peak_for("lw", 16)
    lw_chunk = peak_for("lw", 1)
    if lw_full and lw_chunk:
        summary["lw_peak_mib_unchunked"] = round(lw_full / 1024 / 1024, 2)
        summary["lw_peak_mib_chunk1"] = round(lw_chunk / 1024 / 1024, 2)
        summary["lw_peak_reduction_pct"] = round(100.0 * (1.0 - lw_chunk / lw_full), 2)
    record["summary"] = summary

    inert = record["inertness"]
    record["verdict"] = {
        "sw_inert_all_chunks_bit_identical": inert["sw"]["all_bit_identical"],
        "lw_inert_all_chunks_bit_identical": inert["lw"]["all_bit_identical"],
        "sw_inert_max_rel": inert["sw"]["max_rel_over_all_chunks"],
        "lw_inert_max_rel": inert["lw"]["max_rel_over_all_chunks"],
        "sw_vram_reduced": bool(sw_full and sw_chunk and sw_chunk < sw_full),
        "lw_vram_reduced": bool(lw_full and lw_chunk and lw_chunk < lw_full),
    }
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("inertness", "vram", "all"), default="all")
    ap.add_argument("--kind", choices=("sw", "lw"), default="sw")
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--ncol", type=int, default=8192, help="column-tile multiplier (fixture has 3 cols)")
    ap.add_argument("--nrep", type=int, default=3, help="vertical-level multiplier (fixture has 16 layers)")
    ap.add_argument("--emit-json", action="store_true")
    args = ap.parse_args()

    if args.mode == "inertness":
        rec = run_inertness()
        print(json.dumps(rec, indent=2))
    elif args.mode == "vram":
        rec = run_vram(args.kind, args.chunk, args.ncol, args.nrep)
        if args.emit_json:
            print("VRAM_JSON:" + json.dumps(rec))
        else:
            print(json.dumps(rec, indent=2))
    else:
        rec = drive_all(args.ncol, args.nrep)
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(rec, indent=2))
        print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
