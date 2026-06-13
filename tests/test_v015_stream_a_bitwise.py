"""v0.15 Stream-A bitwise pins (CPU).

The authoritative real-case fp64 gates live in proofs/perf/v015/ab_compare_*.json
(GPU, full 3-hour production program). These tests pin the unit-level contracts:

  1. The Python-unrolled `_condensation_edmf` == the original lax.fori_loop
     formulation (loop peeling does not reassociate).
  2. lax-level: a reverse=True scan == the flip-scan-flip formulation
     (documents the property; the production advance_w keeps the v0.14
     flip formulation because the FULL-program fusion context shifted last
     bits -- see the v0.15 kernel-probe review).

LESSON pinned by the v0.15 probe: unit-level bitwise equality does NOT imply
full-program bitwise equality -- XLA FMA contraction depends on fusion context.
Every Stream-A change must pass the real-case ab hash gate, not just these.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from gpuwrf.dynamics.core.advance_w import advance_w_wrf
from gpuwrf.physics.mynn_edmf import P1000MB, RCP, XLVCP, _condensation_edmf, _qsat_blend

NZ = 44


def _advance_w_kwargs(n=12, seed=7):
    rng = np.random.default_rng(seed)
    f3 = lambda nl: jnp.asarray(1.0 + 0.01 * rng.standard_normal((nl, n, n)))
    f2 = lambda: jnp.asarray(1.0 + 0.01 * rng.standard_normal((n, n)))
    return dict(
        w=f3(NZ + 1), rw_tend=0.001 * f3(NZ + 1), ww=0.01 * f3(NZ + 1),
        u=jnp.asarray(1.0 + 0.01 * rng.standard_normal((NZ, n, n + 1))),
        v=jnp.asarray(1.0 + 0.01 * rng.standard_normal((NZ, n + 1, n))),
        mu_work=0.01 * f2(), mut=1e4 * f2(), muave=0.01 * f2(), muts=1e4 * f2(),
        t_2ave=f3(NZ), t_2=f3(NZ), t_1=f3(NZ),
        ph=f3(NZ + 1), ph_1=f3(NZ + 1), phb=1e3 * f3(NZ + 1), ph_tend=0.001 * f3(NZ + 1),
        ht=100.0 * f2(), c2a=f3(NZ), cqw=f3(NZ), alt=f3(NZ),
        a=0.1 * f3(NZ + 1), alpha=0.9 * f3(NZ + 1), gamma=0.1 * f3(NZ + 1),
        c1h=jnp.asarray(np.linspace(1.0, 0.1, NZ)), c2h=jnp.asarray(np.linspace(0.0, 100.0, NZ)),
        c1f=jnp.asarray(np.linspace(1.0, 0.1, NZ + 1)), c2f=jnp.asarray(np.linspace(0.0, 100.0, NZ + 1)),
        rdnw=jnp.asarray(np.full(NZ, 44.0)), rdn=jnp.asarray(np.full(NZ, 44.0)),
        fnm=jnp.asarray(np.full(NZ, 0.5)), fnp=jnp.asarray(np.full(NZ, 0.5)),
        cf1=jnp.asarray(1.5), cf2=jnp.asarray(-0.5), cf3=jnp.asarray(0.0),
        msftx=f2(), msfty=f2(), w_save=0.01 * f3(NZ + 1),
        rdx=1.0 / 7700.0, rdy=1.0 / 7700.0, dts=1.8, epssm=0.5, top_lid=True,
        damp_opt=3, dampcoef=0.2, zdamp=5000.0, w_damping=1,
    )


def test_advance_w_thomas_unroll_env_default_inert() -> None:
    """GPUWRF_THOMAS_UNROLL default (1) must lower exactly like v0.14 (False)."""
    kw = _advance_w_kwargs()
    ref = advance_w_wrf(**kw)  # default env -> unroll=False lowering
    import os

    os.environ["GPUWRF_THOMAS_UNROLL"] = "45"
    try:
        out = advance_w_wrf(**kw)
    finally:
        os.environ.pop("GPUWRF_THOMAS_UNROLL", None)
    for r, o in zip(ref, out):
        assert (np.asarray(r) == np.asarray(o)).all()


def test_thomas_reverse_scan_matches_flip_formulation() -> None:
    rng = np.random.default_rng(0)
    n = 8
    gamma = jnp.asarray(0.1 * rng.standard_normal((NZ + 1, n, n)))
    w_fwd = jnp.asarray(rng.standard_normal((NZ + 1, n, n)))

    def _back(next_w, e):
        g, wk = e
        out = wk - g * next_w
        return out, out

    gr = gamma[1:NZ][::-1]
    wr = w_fwd[1:NZ][::-1]
    _, rev = lax.scan(_back, w_fwd[NZ], (gr, wr), unroll=False)
    old = rev[::-1]
    _, new = lax.scan(_back, w_fwd[NZ], (gamma[1:NZ], w_fwd[1:NZ]), unroll=1, reverse=True)
    assert (np.asarray(old) == np.asarray(new)).all()


def test_condensation_unroll_matches_fori() -> None:
    """Python-unrolled condensation == the production fori formulation (bitwise).

    The unroll was proven bit-identical IN THE FULL PROGRAM
    (ab_compare_v014_base_vs_streamA_final.json, 0/168 mismatches) and collapses
    device kernel time 99.4 -> 15.4 ms/step, but was reverted from production
    because the untraced wall regressed +7.5 % (host submit path; see the v0.15
    kernel-probe review). This pin preserves the proven equivalence for the
    v0.15-S1 re-landing (unroll + CUDA-graph capture together).
    """
    rng = np.random.default_rng(3)
    n = 512
    qt = jnp.asarray(0.002 + 0.01 * rng.random(n))
    thl = jnp.asarray(280.0 + 20.0 * rng.random(n))
    p = jnp.asarray(5e4 + 5e4 * rng.random(n))
    z = jnp.asarray(2000.0 * rng.random(n))

    def unrolled(qt, thl, p, zagl, niter=50):
        exn = (p / P1000MB) ** RCP
        qc = jnp.zeros_like(qt)
        for _ in range(int(niter)):
            t = exn * thl + XLVCP * qc
            qs = _qsat_blend(t, p)
            qc = 0.5 * qc + 0.5 * jnp.maximum(qt - qs, 0.0)
        t = exn * thl + XLVCP * qc
        qs = _qsat_blend(t, p)
        qc = jnp.maximum(qt - qs, 0.0)
        return jnp.where(zagl < 100.0, 0.0, qc)

    qc_unrolled = jax.jit(unrolled)(qt, thl, p, z)
    _, qc_prod = jax.jit(_condensation_edmf)(qt, thl, p, z)  # production fori
    assert (np.asarray(qc_unrolled) == np.asarray(qc_prod)).all()
