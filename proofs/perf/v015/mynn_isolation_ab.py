"""v0.15 MYNN PBL ISOLATION A/B -- the dominant 88 ms/step kernel (NVTX-attributed).

The Fable nsys NVTX attribution shows MYNN_PBL(EDMF) = 88.6 ms/step (87% of the
178 ms d01 step); the dycore is only ~10 ms. So the fp32-ACOUSTIC lead is the wrong
lever -- the win (or the wall) is MYNN. The cost is the dense (B, nz, nz) BouLac
free-atmosphere length-scale parcel search (mynn_pbl._boulac_length): at B=16384
(d01 128x128), nz=44, one such fp64 array = 254 MB and ~20 are live -> ~5 GB
transient HBM/call. Default GPUWRF_MYNN_COLUMN_TILE_COLS=16384 == the d01 batch,
so the production path is UNTILED for d01 (tiling needs B > tile).

This harness isolates step_mynn_pbl_column on the exact (16384, 44) shape and
measures two identity-preserving levers:
  (A) COLUMN TILING width {16384(off), 8192, 4096, 2048, 1024} -- proven
      GPU-bit-identical (module note); does shrinking the dense working set to
      fit cache cut wall time?  (warmed wall + GPU bit-identity vs untiled)
  (B) reserved for fp32 BouLac (separate harness mynn_fp32_boulac_ab.py).

Run (GPU lock required):
  scripts/with_gpu_lock.sh --label perf-fix -- \
    taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
      python proofs/perf/v015/mynn_isolation_ab.py
"""
from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

OUT = Path("proofs/perf/v015")
B, NZ = 16384, 44   # d01 128x128 columns, 44 mass levels
DT = 18.0
DX = 3000.0


def _block(x):
    jax.block_until_ready(x)
    return x


def _make_state(mod, dtype):
    """A physically-plausible (B, nz) MYNN column batch (shape drives BouLac cost)."""
    rng = np.random.default_rng(20260612)
    z = np.cumsum(np.full((B, NZ), 250.0, np.float64), axis=-1)  # ~250 m layers
    th = 290.0 + 0.004 * z + rng.normal(0, 0.3, (B, NZ))         # stable-ish lapse + noise
    qv = np.clip(0.008 - 1.0e-6 * z + rng.normal(0, 2e-4, (B, NZ)), 1e-6, None)
    u = 5.0 + 0.002 * z + rng.normal(0, 1.0, (B, NZ))
    v = rng.normal(0, 1.0, (B, NZ))
    w = np.zeros((B, NZ))
    p = 1.0e5 * np.exp(-z / 8500.0)
    rho = p / (287.0 * th)
    dz = np.full((B, NZ), 250.0, np.float64)
    tke = np.clip(0.5 + rng.normal(0, 0.1, (B, NZ)), 0.02, None)
    km = np.full((B, NZ), 1.0); kh = np.full((B, NZ), 1.0); el = np.full((B, NZ), 50.0)
    qc = np.zeros((B, NZ)); qi = np.zeros((B, NZ))

    def a(x):
        return jnp.asarray(x, dtype=dtype)

    return mod.MynnPBLColumnState(
        u=a(u), v=a(v), w=a(w), theta=a(th), qv=a(qv), tke=a(tke), p=a(p),
        rho=a(rho), dz=a(dz), km=a(km), kh=a(kh), el=a(el), qc=a(qc), qi=a(qi),
    )


def _reload_mynn(tile_cols: int):
    """Reimport mynn_pbl with the env tile width baked into module constants."""
    os.environ["GPUWRF_MYNN_COLUMN_TILE_COLS"] = str(tile_cols)
    os.environ["GPUWRF_MYNN_COLUMN_TILING"] = "0" if tile_cols == 0 else "1"
    import gpuwrf.physics.mynn_pbl as m
    importlib.reload(m)
    return m


def _time_call(fn, state, reps=6):
    _block(fn(state, DT, debug=False, surface=None, edmf=True, dx=DX))  # compile
    _block(fn(state, DT, debug=False, surface=None, edmf=True, dx=DX))  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(state, DT, debug=False, surface=None, edmf=True, dx=DX)
        _block(out)
        ts.append(time.perf_counter() - t0)
    return out, ts


def main() -> int:
    results = {}
    ref_out = None
    # Untiled fp64 reference (the production d01 path: tile==B -> no tiling).
    for tag, tile in [("untiled_16384", 16384), ("tile_8192", 8192),
                      ("tile_4096", 4096), ("tile_2048", 2048), ("tile_1024", 1024)]:
        m = _reload_mynn(tile)
        st = _make_state(m, jnp.float64)
        out, ts = _time_call(m.step_mynn_pbl_column, st)
        ms = min(ts) * 1000.0
        # GPU bit-identity vs the untiled reference (theta is the prognostic out).
        identical = None
        maxdiff = None
        out_np = np.asarray(out.theta)
        if ref_out is None:
            ref_out = out_np
        else:
            maxdiff = float(np.max(np.abs(out_np - ref_out)))
            identical = bool(maxdiff == 0.0)
        results[tag] = {
            "tile_cols": tile, "min_ms": ms, "samples_ms": [t * 1000 for t in ts],
            "bit_identical_vs_untiled": identical, "max_abs_theta_diff_vs_untiled": maxdiff,
        }
        print(f"{tag:16s} tile={tile:6d} min_ms={ms:8.2f} "
              f"identical={identical} maxdiff={maxdiff}", flush=True)

    base = results["untiled_16384"]["min_ms"]
    for tag, r in results.items():
        r["speedup_vs_untiled"] = base / r["min_ms"] if r["min_ms"] else None

    out = {
        "scope": "v0.15 MYNN PBL isolation: column-tiling A/B on the dominant 88ms/step kernel",
        "device": str(jax.devices()[0]),
        "shape": {"B": B, "nz": NZ}, "dt_s": DT, "dx_m": DX, "edmf": True,
        "note": "step_mynn_pbl_column isolated, fp64; tiling proven GPU-bit-identical (module note). "
                "Production d01 path = untiled_16384 (tile==B).",
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / "mynn_isolation_ab.json"
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {fn}", flush=True)
    print(json.dumps({k: {"min_ms": v["min_ms"], "speedup": v["speedup_vs_untiled"],
                          "identical": v["bit_identical_vs_untiled"]}
                      for k, v in results.items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
