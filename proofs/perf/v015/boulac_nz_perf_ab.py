"""v0.15 BouLac O(nz) perf + VRAM A/B -- dense (B,nz,nz) vs lax.scan O(nz).

Isolates ``step_mynn_pbl_column`` on the exact d01 (16384, 44) production shape
and measures the wall + peak-VRAM delta of the BouLac length-scale change:
  - DENSE: the frozen pre-O(nz) ``_boulac_length`` (proofs/perf/v015/
    _boulac_dense_frozen.py, commit 0b2a7066) monkeypatched into mynn_pbl.
  - SCAN:  the production O(nz) ``_boulac_length`` (current branch).

Both are run through the SAME ``step_mynn_pbl_column`` so the only difference is
the length-scale algorithm.  Peak VRAM is read from the JAX device
``memory_stats()`` ``peak_bytes_in_use`` around each compiled call.

Each variant runs in its OWN process (peak_bytes_in_use is process-monotonic),
writing a per-variant json; a final merge step combines them.  Run all three
under ONE GPU lock so timing is not perturbed by an interleaved sibling:

  scripts/with_gpu_lock.sh --label v015-boulac -- bash -c '
    for V in dense scan; do
      taskset -c 0-3 env PYTHONPATH=proofs/perf/v015:src JAX_ENABLE_X64=true \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
        XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
        GPUWRF_JAX_CACHE=0 python proofs/perf/v015/boulac_nz_perf_ab.py --variant $V
    done
    taskset -c 0-3 env PYTHONPATH=proofs/perf/v015:src python \
      proofs/perf/v015/boulac_nz_perf_ab.py --merge'
"""
from __future__ import annotations

import argparse
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


def _peak_vram_bytes(dev):
    try:
        return int(dev.memory_stats().get("peak_bytes_in_use", 0))
    except Exception:
        return 0


def _reset_peak(dev):
    # memory_stats peak is monotonic per-process; re-import does not reset it, so
    # we record the running peak before/after each variant and report the delta.
    return _peak_vram_bytes(dev)


def _time_call(fn, state, reps=8):
    _block(fn(state, DT, debug=False, surface=None, edmf=True, dx=DX))  # compile
    _block(fn(state, DT, debug=False, surface=None, edmf=True, dx=DX))  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(state, DT, debug=False, surface=None, edmf=True, dx=DX)
        _block(out)
        ts.append(time.perf_counter() - t0)
    return out, ts


def _run_variant(use_dense: bool, dev):
    # fresh module each variant; production tiling defaults (untiled at d01).
    os.environ.pop("GPUWRF_MYNN_COLUMN_TILE_COLS", None)
    os.environ.pop("GPUWRF_MYNN_COLUMN_TILING", None)
    os.environ["GPUWRF_MYNN_BOULAC_FP32"] = "0"
    import gpuwrf.physics.mynn_pbl as m
    importlib.reload(m)
    label = "scan_onz"
    if use_dense:
        import _boulac_dense_frozen as df
        importlib.reload(df)
        m._boulac_length = df._boulac_length
        label = "dense_nznz"
    st = _make_state(m)
    peak_before = _peak_vram_bytes(dev)
    out, ts = _time_call(m.step_mynn_pbl_column, st)
    peak_after = _peak_vram_bytes(dev)
    ms = min(ts) * 1000.0
    return {
        "label": label,
        "min_ms": ms,
        "samples_ms": [t * 1000 for t in ts],
        "peak_vram_bytes_before": peak_before,
        "peak_vram_bytes_after": peak_after,
        "peak_vram_delta_mb": (peak_after - peak_before) / 1e6,
        "out_theta": np.asarray(out.theta),
        "out_el": np.asarray(out.el),
        "out_kh": np.asarray(out.kh),
        "out_tke": np.asarray(out.tke),
    }


def _rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    den = np.maximum(np.abs(a), 1e-30)
    return {
        "max_abs": float(np.max(np.abs(a - b))),
        "max_rel": float(np.max(np.abs(a - b) / den)),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
    }


def _variant_json(variant):
    return OUT / f"boulac_nz_perf_{variant}.json"


def run_variant(variant: str) -> int:
    dev = jax.devices()[0]
    r = _run_variant(use_dense=(variant == "dense"), dev=dev)
    # dump prognostic/diagnostic outputs for the cross-variant identity check.
    np.savez_compressed(
        OUT / f"boulac_nz_perf_{variant}_out.npz",
        theta=r.pop("out_theta"), el=r.pop("out_el"),
        kh=r.pop("out_kh"), tke=r.pop("out_tke"),
    )
    r["device"] = str(dev)
    r["shape"] = {"B": B, "nz": NZ}
    _variant_json(variant).write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({"variant": variant, "min_ms": r["min_ms"],
                      "peak_vram_after_mb": r["peak_vram_bytes_after"] / 1e6}, indent=2),
          flush=True)
    return 0


def merge() -> int:
    dense = json.loads(_variant_json("dense").read_text())
    scan = json.loads(_variant_json("scan").read_text())
    od = np.load(OUT / "boulac_nz_perf_dense_out.npz")
    os_ = np.load(OUT / "boulac_nz_perf_scan_out.npz")
    identity = {k: _rel(od[k], os_[k]) for k in ("theta", "el", "kh", "tke")}

    payload = {
        "scope": "v0.15 BouLac O(nz) perf+VRAM A/B (isolated step_mynn_pbl_column, d01 16384x44)",
        "device": dense.get("device"),
        "shape": {"B": B, "nz": NZ}, "dt_s": DT, "dx_m": DX, "edmf": True,
        "dense_nznz": dense,
        "scan_onz": scan,
        "mynn_step_speedup_scan_vs_dense": dense["min_ms"] / scan["min_ms"] if scan["min_ms"] else None,
        "mynn_step_saved_ms": dense["min_ms"] - scan["min_ms"],
        "peak_vram_dense_path_mb": dense["peak_vram_bytes_after"] / 1e6,
        "peak_vram_scan_path_mb": scan["peak_vram_bytes_after"] / 1e6,
        "peak_vram_saved_mb": (dense["peak_vram_bytes_after"] - scan["peak_vram_bytes_after"]) / 1e6,
        "scan_vs_dense_step_output_identity": identity,
        "note": "Each variant in its OWN process (peak_bytes_in_use is "
                "process-monotonic). Isolated MYNN step; full-pipeline wall is in "
                "the tiered-gate probe steady_ms_per_step. The scan path HLO has 0 "
                "dense (B,nz,nz) tensors (110 occurrences in the dense path).",
    }
    fn = OUT / "boulac_nz_perf_ab.json"
    fn.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "dense_ms": dense["min_ms"], "scan_ms": scan["min_ms"],
        "mynn_step_speedup": payload["mynn_step_speedup_scan_vs_dense"],
        "saved_ms": payload["mynn_step_saved_ms"],
        "peak_vram_dense_mb": payload["peak_vram_dense_path_mb"],
        "peak_vram_scan_mb": payload["peak_vram_scan_path_mb"],
        "peak_vram_saved_mb": payload["peak_vram_saved_mb"],
        "theta_identity_max_rel": identity["theta"]["max_rel"],
        "el_identity_max_rel": identity["el"]["max_rel"],
    }, indent=2), flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["dense", "scan"])
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge()
    if args.variant:
        return run_variant(args.variant)
    ap.error("specify --variant {dense,scan} or --merge")


if __name__ == "__main__":
    raise SystemExit(main())
