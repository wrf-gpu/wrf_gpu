"""Wave-B2 STEP 2 probe: EDMF vertical-integration scan unroll A/B (round-off-NEUTRAL codegen only).

The MYNN closure kernel (32 ms) is 94% of the surface+MYNN block; the EDMF
mass-flux phase is its single largest internal component (~15.7 ms). EDMF is a
2-level vmap (columns x plumes) of a lax.scan over ~nz-2 vertical levels. A scan
`unroll` is pure XLA codegen -- it does NOT change the math or the iteration
count, so it is round-off-NEUTRAL (the same class Wave-A proved bit-identical for
the acoustic substep scan). This probes whether unrolling the EDMF level scan
reduces launch overhead on the coupled warmed step WITHOUT any fidelity change.

If the warmed gain is noise (<~1%) or it is NOT bit-identical, REJECT (do not
touch the closure).
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.config import paths
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _enforce_operational_precision
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.physics import mynn_edmf

PROOF = Path("proofs/v0100")
L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"
DT = 10.0


def _bench(fn, *args, n_rep=12, label=""):
    out = fn(*args)
    jax.block_until_ready(out)
    samples = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"label": label, "min_ms": float(min(samples)),
            "median_ms": float(np.median(samples)),
            "samples_ms": [round(s, 4) for s in samples]}


def main() -> int:
    cfg = DailyPipelineConfig(hours=1, dt_s=DT, acoustic_substeps=10,
                             run_id=L2_RUN_ID, run_root=paths.wrf_l2_root(),
                             domain="d02", radiation_cadence_steps=180)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                             disable_guards=False, radiation_cadence_steps=180,
                             time_utc=case.run_start)
    grid = nl.grid
    state = _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    state = jax.tree_util.tree_map(lambda a: jnp.asarray(a) if hasattr(a, "shape") else a, state)
    state_with_flux = pc.surface_adapter(state, DT)

    mynn_jit = jax.jit(lambda s: pc.mynn_adapter(s, DT, grid))

    # Baseline (scan as shipped).
    orig_scan = jax.lax.scan
    res_base = _bench(mynn_jit, state_with_flux, label="edmf_scan_default")
    base_out = mynn_jit(state_with_flux)

    # Monkeypatch lax.scan inside mynn_edmf to add unroll on the level scan.
    UNROLL = 4
    def _patched_scan(f, init, xs=None, length=None, **kw):
        if "unroll" not in kw:
            kw["unroll"] = UNROLL
        return orig_scan(f, init, xs=xs, length=length, **kw)

    mynn_edmf.lax.scan = _patched_scan
    try:
        # New jit captures the patched scan.
        mynn_jit2 = jax.jit(lambda s: pc.mynn_adapter(s, DT, grid))
        res_unroll = _bench(mynn_jit2, state_with_flux, label=f"edmf_scan_unroll{UNROLL}")
        unroll_out = mynn_jit2(state_with_flux)
    finally:
        mynn_edmf.lax.scan = orig_scan

    fields = ("u", "v", "theta", "qv", "qke")
    diffs = {}
    bit_identical = True
    for f in fields:
        a = np.asarray(jax.device_get(getattr(base_out, f)), dtype=np.float64)
        b = np.asarray(jax.device_get(getattr(unroll_out, f)), dtype=np.float64)
        equal = bool(np.array_equal(a, b))
        denom = np.maximum(np.abs(a), 1e-30)
        reldiff = float(np.nanmax(np.abs(a - b) / denom)) if a.size else 0.0
        diffs[f] = {"array_equal": equal, "max_reldiff": reldiff}
        bit_identical = bit_identical and equal

    gain_ms = res_base["min_ms"] - res_unroll["min_ms"]
    gain_pct = 100.0 * gain_ms / res_base["min_ms"] if res_base["min_ms"] else 0.0

    out = {
        "scope": f"Wave-B2 EDMF level-scan unroll={UNROLL} A/B (round-off-neutral codegen)",
        "run_dir": str(run_dir), "device": str(jax.devices()[0]),
        "unroll": UNROLL,
        "mynn_default_min_ms": res_base["min_ms"],
        "mynn_unroll_min_ms": res_unroll["min_ms"],
        "default_samples_ms": res_base["samples_ms"],
        "unroll_samples_ms": res_unroll["samples_ms"],
        "block_gain_ms": gain_ms, "block_gain_pct": gain_pct,
        "bit_identical": bit_identical, "field_diffs": diffs,
        "verdict": ("KEEP" if (bit_identical and gain_pct > 2.0) else "REJECT"),
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / "wave_b2_edmf_unroll_ab.json"
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in (
        "mynn_default_min_ms", "mynn_unroll_min_ms", "block_gain_ms",
        "block_gain_pct", "bit_identical", "verdict")}, indent=2), flush=True)
    print(f"wrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
