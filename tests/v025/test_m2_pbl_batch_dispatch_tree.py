"""M2 family-#2 G0: the 672-dispatch structure of ``jit(step_mynn_pbl_column)``.

W2 measured ``loop_multiply_fusion_32`` = 672 launches/step @ 11.2 us (52.2% of
device time) attributed by W2.1 HLO metadata to ``jit(step_mynn_pbl_column)``
with kernel shape f64[8400, 8].  This test pins the SOURCE structure that
produces those launches, on CPU, from the live code (not from the sprint
report's claim):

    dmp_mf_columns                    mynn_edmf.py:554
    └─ vmap over columns (B)          mynn_edmf.py:566        (vmap #1 -> dim 8400)
       └─ vmap over NUP=8 plumes      mynn_edmf.py:436        (vmap #2 -> dim 8)
          └─ lax.scan over nz-2 lvls  mynn_edmf.py:425        (WHILE #1, trip = 42 @ nz=44)
             └─ _condensation_edmf    mynn_edmf.py:171,381
                └─ lax.fori_loop(16)  mynn_edmf.py:197        (WHILE #2, trip = 16)

    42 x 16 = 672 sequential innermost iterations per timestep.

The production case (W2 FAST arm d01) has e_vert=45 => nz=44 mass levels =>
42 scanned levels; GPUWRF_MYNN_COND_NITER defaults to 16.  The HLO census here
asserts exactly the two trip counts the W2.1 GPU dump printed for the same
code (backend_config known_trip_count 42 / 16).
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.pop("GPUWRF_MYNN_COND_NITER", None)  # pin the default 16
os.environ.pop("GPUWRF_MYNN_COND_UNROLL", None)  # pin the default fori lowering
os.environ.pop("GPUWRF_EDMF_FUSED_PLUME", None)  # reference path

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from gpuwrf.physics import mynn_edmf  # noqa: E402

NZ = 44  # W2 FAST arm d01: e_vert=45 => 44 mass levels
B = 4  # HLO structure is B-independent (vmap width is metadata)


def _fixture(batch: int = B, nz: int = NZ):
    prof = lambda fill, shift=0.0: jnp.asarray(
        fill + shift * jnp.arange(batch * nz, dtype=jnp.float64).reshape(batch, nz)
    )
    col = lambda fill: jnp.full((batch,), fill, jnp.float64)
    zw = jnp.cumsum(prof(30.0), axis=-1)
    zw = jnp.concatenate([jnp.zeros((batch, 1)), zw], axis=-1)
    return dict(
        sqw=prof(0.010), sqv=prof(0.009), sqc=prof(0.0001),
        u=prof(3.0), v=prof(1.0), w=prof(0.5),
        th=prof(300.0), thl=prof(299.0), thv=prof(301.0),
        tk=prof(280.0), qke=prof(0.4),
        p=prof(90000.0, -3.0), exner=prof(0.95),
        rho=prof(1.1), dz=prof(30.0), zw=zw,
        ust=col(0.2), flt=col(0.05), fltv=col(0.06), flq=col(1e-5),
        flqv=col(1e-5), pblh=col(900.0), ts=col(290.0),
        dx=1000.0, xland=col(1.0), dt=10.0,
    )


def _hlo_text() -> str:
    args = _fixture()
    lowered = jax.jit(mynn_edmf.dmp_mf_columns).lower(**args)
    return lowered.as_text()  # stablehlo MLIR text (CPU backend)


def test_672_equals_levels_times_cond_iterations() -> None:
    """The arithmetic identity behind the measured 672/step."""
    nup = mynn_edmf.NUP
    niter = mynn_edmf._cond_niter()
    levels = NZ - 2  # ks = jnp.arange(1, nz-1) at mynn_edmf.py:421
    assert (nup, niter) == (8, 16)
    assert levels == 42
    assert levels * niter == 672
    # W2 cross-check: 725,760 captured instances / 1080 captured steps = 672.
    assert 725_760 == 672 * 1080


def test_hlo_shows_the_two_nested_while_trip_counts() -> None:
    """WHILE #1 (levels, trip=nz-2) and WHILE #2 (cond fixed point, trip=16).

    The W2.1 GPU dump printed these as backend_config
    ``known_trip_count 42 / 16`` on the same source; the CPU backend lowers to
    the same two whiles (stablehlo, no trip annotation), so here we pin the
    counts at the backend-independent jaxpr level (see
    :func:`test_condensation_loop_is_inside_the_level_loop`) and census the
    while structure in the compiled MLIR text.
    """
    text = _hlo_text()
    # Exactly the two sequential whiles: plume-level scan + cond fixed point.
    n_whiles = len(re.findall(r"stablehlo\.while\(", text))
    assert n_whiles == 2, f"expected exactly 2 whiles (level scan + cond fori); got {n_whiles}"


def _cond_literals(jaxpr) -> set[int]:
    out = set()
    for eqn in jaxpr.eqns:
        for v in eqn.invars:
            val = getattr(v, "val", None)
            if isinstance(val, int):
                out.add(val)
    return out


def test_condensation_loop_is_inside_the_level_loop() -> None:
    """Nesting + trip-count proof at backend-independent jaxpr level.

    The 16-iteration fixed point (from ``_condensation_edmf``'s
    ``lax.fori_loop``, mynn_edmf.py:197, reached from the scan step body at
    mynn_edmf.py:381) must sit INSIDE the plume-level scan's body jaxpr
    (mynn_edmf.py:425).  A sibling (post-scan) lowering would show the loop
    OUTSIDE the scan subjaxpr instead.  ``length == nz-2`` on the outer scan
    pins the 42; the inner loop must carry trip COND_NITER = 16.  In the jaxpr
    the inner loop appears as a ``while`` (direct fori trace) or as a nested
    ``scan`` (while batched by the plume vmap) -- both lower to the HLO while
    the W2.1 GPU dump annotated ``known_trip_count=16``.
    """
    args = _fixture()
    jaxpr = jax.make_jaxpr(mynn_edmf.dmp_mf_columns, return_shape=True)(
        **args
    )[0]
    scan_eqns = [e for e in jaxpr.eqns if e.primitive.name == "scan"]
    assert len(scan_eqns) == 1, "expected exactly the plume-rise scan"
    scan_eqn = scan_eqns[0]
    params = scan_eqn.params
    assert int(params["length"]) == NZ - 2
    body = params["jaxpr"]  # scan body: step() incl. _condensation_edmf
    inner_names = [e.primitive.name for e in body.eqns]
    loop_eqns = [e for e in body.eqns if e.primitive.name in ("while", "scan")]
    assert loop_eqns, (
        "condensation fixed point must be a sequential loop INSIDE the scan "
        f"body; got {inner_names}"
    )
    trip = None
    for e in loop_eqns:
        if e.primitive.name == "scan":
            trip = int(e.params["length"])
        else:
            cond_jaxpr = e.params["cond_jaxpr"]
            cond_jaxpr = getattr(cond_jaxpr, "jaxpr", cond_jaxpr)
            assert 16 in _cond_literals(cond_jaxpr), (
                "condensation while bound must be the COND_NITER literal 16"
            )
            trip = 16
    assert trip == 16, f"condensation loop trip must be COND_NITER=16; got {trip}"
