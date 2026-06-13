#!/usr/bin/env python
"""v0.15 S1 — condensation niter-cap oracle (tier-1 identity evidence).

Bounds the numerics change of GPUWRF_MYNN_COND_NITER=16 vs the v0.14 default 50
on the REAL `_condensation_edmf` over a dense sweep of the full physical input
envelope the EDMF plume scan can produce (qt up to 30 g/kg, thl 250-330 K,
p 1050-100 hPa, zagl above the 100 m gate).  WRF itself exits this loop when
|QC-QCold| < diff and "usually converges in < 8 iterations"
(module_bl_mynnedmf.F:6794-6851); the iteration is a 0.5-damped fixed point, so
the residual after n iters contracts ~0.5^n.  niter=16 leaves ~0.5^16 = 1.5e-5
of the initial gap -- this oracle MEASURES the worst case over the envelope.

Also proves the lowering equivalence: unrolled@16 == fori@16 bitwise on GPU
(same op sequence, loop peeling only).

Artifact: proofs/perf/v015/cond_niter_oracle.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent

# Import with default env so module knobs are inert; niter passed explicitly.
os.environ.setdefault("GPUWRF_MYNN_COND_NITER", "50")
os.environ.setdefault("GPUWRF_MYNN_COND_UNROLL", "0")
from gpuwrf.physics import mynn_edmf as me  # noqa: E402


def run(niter: int, unroll: bool):
    qt = jnp.linspace(0.0, 0.030, 121, dtype=jnp.float64)        # kg/kg
    thl = jnp.linspace(250.0, 330.0, 81, dtype=jnp.float64)      # K
    p = jnp.linspace(1.05e5, 1.0e4, 41, dtype=jnp.float64)       # Pa
    QT, THL, P = jnp.meshgrid(qt, thl, p, indexing="ij")
    Z = jnp.full_like(QT, 500.0)  # above the 100 m zero gate

    def fn(qt_, thl_, p_, z_):
        if unroll:
            os_val = os.environ.get("GPUWRF_MYNN_COND_UNROLL")
            os.environ["GPUWRF_MYNN_COND_UNROLL"] = "1"
            try:
                out = me._condensation_edmf(qt_, thl_, p_, z_, niter=niter)
            finally:
                os.environ["GPUWRF_MYNN_COND_UNROLL"] = os_val
            return out
        return me._condensation_edmf(qt_, thl_, p_, z_, niter=niter)

    thv, qc = jax.jit(lambda a, b, c, d: fn(a, b, c, d))(QT, THL, P, Z)
    return jax.device_get(thv), jax.device_get(qc)


def main() -> int:
    import numpy as np

    thv50, qc50 = run(50, unroll=False)   # v0.14 production numerics
    thv16, qc16 = run(16, unroll=False)   # capped, same lowering
    thv16u, qc16u = run(16, unroll=True)  # capped, unrolled lowering
    thv8, qc8 = run(8, unroll=False)      # WRF's "usually converges" bound

    def stats(a, b):
        d = np.abs(a - b)
        rel = d / np.maximum(np.abs(b), 1e-300)
        return {
            "max_abs": float(d.max()),
            "max_rel_where_b_gt_1e-12": float(np.where(np.abs(b) > 1e-12, rel, 0.0).max()),
            "n_diff": int((d > 0).sum()),
            "n_total": int(d.size),
        }

    payload = {
        "schema": "V015CondNiterOracle",
        "envelope": "qt 0-30 g/kg x thl 250-330 K x p 100-1050 hPa (401,841 states)",
        "qc16_vs_qc50": stats(qc16, qc50),
        "thv16_vs_thv50": stats(thv16, thv50),
        "qc8_vs_qc50": stats(qc8, qc50),
        "qc16_unrolled_vs_fori_bitwise": bool((qc16u == qc16).all() and (thv16u == thv16).all()),
        "qc_scale_typical_kgkg": 1e-3,
        "note": (
            "MEASURED VERDICT (2026-06-12): in the CONVERGENT regime the 16-iter "
            "residual is < WRF's own exit threshold diff=1e-6 (unit sweep max "
            "1.6e-7). The envelope's warm+very-moist corner is NON-convergent for "
            "the WRF iteration itself (|lambda|>1: qs-feedback gain exceeds the "
            "0.5 damping): there qc oscillates and qc50 vs qc16 are different "
            "phase samples (max_abs ~4e-3) -- WRF's own 50th iterate is equally "
            "arbitrary there (its early-exit never trips). Consequence: the "
            "niter cap is justified analytically ONLY in the convergent regime; "
            "the production decision rests on the Tier-P field gate "
            "(compare_tiered_identity vs the frozen v0.14 manifest). The "
            "unrolled==fori bitwise bit proves the lowering knob alone changes "
            "nothing at equal niter."
        ),
    }
    out = HERE / "cond_niter_oracle.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
