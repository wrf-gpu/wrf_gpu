"""v0.15 MYNN fp32-BouLac A/B -- halve the dense (B,nz,nz) length-scale traffic.

Engages GPUWRF_MYNN_BOULAC_FP32 (added to mynn_pbl._boulac_length, default OFF):
the dense parcel search runs in fp32, lengths cast back to fp64.  Measures the
isolated step_mynn_pbl_column wall fp64-BouLac vs fp32-BouLac on the exact d01
(16384,44) shape, and the IDENTITY of the change on the prognostic outputs
(theta/qv/tke) and the diagnostic el/km/kh -- this is the operational-impact
number for the frozen-tolerance gate (per-field downcast gated by op-impact, not
bitwise).

Run (GPU lock required):
  scripts/with_gpu_lock.sh --label perf-fix -- \
    taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
      python proofs/perf/v015/mynn_fp32_boulac_ab.py
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
B, NZ = 16384, 44
DT = 18.0
DX = 3000.0


def _block(x):
    jax.block_until_ready(x)
    return x


def _make_state(mod, dtype=jnp.float64):
    rng = np.random.default_rng(20260612)
    z = np.cumsum(np.full((B, NZ), 250.0, np.float64), axis=-1)
    th = 290.0 + 0.004 * z + rng.normal(0, 0.3, (B, NZ))
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


def _reload(boulac_fp32: bool):
    os.environ["GPUWRF_MYNN_BOULAC_FP32"] = "1" if boulac_fp32 else "0"
    # keep tiling at production default for this isolation
    os.environ.pop("GPUWRF_MYNN_COLUMN_TILE_COLS", None)
    os.environ.pop("GPUWRF_MYNN_COLUMN_TILING", None)
    import gpuwrf.physics.mynn_pbl as m
    importlib.reload(m)
    return m


def _time(fn, st, reps=6):
    _block(fn(st, DT, debug=False, surface=None, edmf=True, dx=DX))
    _block(fn(st, DT, debug=False, surface=None, edmf=True, dx=DX))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(st, DT, debug=False, surface=None, edmf=True, dx=DX)
        _block(out)
        ts.append(time.perf_counter() - t0)
    return out, ts


def _rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    den = np.maximum(np.abs(a), 1e-30)
    return {
        "max_abs": float(np.max(np.abs(a - b))),
        "max_rel": float(np.max(np.abs(a - b) / den)),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
    }


def main() -> int:
    m0 = _reload(False)
    st0 = _make_state(m0)
    out_fp64, ts_fp64 = _time(m0.step_mynn_pbl_column, st0)
    ms_fp64 = min(ts_fp64) * 1000.0

    m1 = _reload(True)
    st1 = _make_state(m1)
    out_fp32, ts_fp32 = _time(m1.step_mynn_pbl_column, st1)
    ms_fp32 = min(ts_fp32) * 1000.0

    # identity vs fp64-BouLac reference on the SAME inputs.
    identity = {
        "theta": _rel(out_fp64.theta, out_fp32.theta),
        "qv": _rel(out_fp64.qv, out_fp32.qv),
        "tke": _rel(out_fp64.tke, out_fp32.tke),
        "u": _rel(out_fp64.u, out_fp32.u),
        "v": _rel(out_fp64.v, out_fp32.v),
        "el": _rel(out_fp64.el, out_fp32.el),
        "km": _rel(out_fp64.km, out_fp32.km),
        "kh": _rel(out_fp64.kh, out_fp32.kh),
    }

    out = {
        "scope": "v0.15 MYNN fp32-BouLac A/B (1 isolated step, d01 16384x44 shape)",
        "device": str(jax.devices()[0]),
        "shape": {"B": B, "nz": NZ}, "dt_s": DT, "dx_m": DX, "edmf": True,
        "fp64_boulac_min_ms": ms_fp64,
        "fp32_boulac_min_ms": ms_fp32,
        "speedup": ms_fp64 / ms_fp32 if ms_fp32 else None,
        "samples_fp64_ms": [t * 1000 for t in ts_fp64],
        "samples_fp32_ms": [t * 1000 for t in ts_fp32],
        "identity_fp32_vs_fp64": identity,
        "note": "1-step isolation identity is the per-step injection; the OPERATIONAL "
                "gate is a short-forecast field-tolerance run (escalate if 1-step rel is small).",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / "mynn_fp32_boulac_ab.json"
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "fp64_ms": ms_fp64, "fp32_ms": ms_fp32, "speedup": out["speedup"],
        "theta_max_rel": identity["theta"]["max_rel"],
        "tke_max_rel": identity["tke"]["max_rel"],
        "el_max_rel": identity["el"]["max_rel"],
        "km_max_rel": identity["km"]["max_rel"],
    }, indent=2), flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
