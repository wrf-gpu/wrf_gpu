"""Fused Pallas kernel for the MYNN-EDMF plume-rise nest (M2 family #2).

Sprint ``2026-09-18-v0250-m2-pbl-batch`` (worker/glm/pblbatch).  This module
replaces the single hottest dispatch family in the product (W2/W2.1,
definitive): the MYNN-EDMF per-plume vertical integration
``mynn_edmf._dmp_scan`` -- a ``lax.scan`` over the 42 scanned levels whose step
body contains the 16-iteration condensation fixed point
(``_condensation_edmf``, ``lax.fori_loop``).  Under XLA that nest dispatches
the fused kernel ``loop_multiply_fusion_32`` **672x per timestep @ 11.2 us =
52.2% of ALL device time** (kernel shape f64[8400,8]: 8400 columns x NUP=8
plumes), and every launch depends on its predecessor (the level chain runs
through the plume buoyancy), so none of the latency overlaps.

Structure being collapsed (file:line in ``mynn_edmf.py``)::

    dmp_mf_columns:554 -> vmap over B columns:566   (vmap #1)
      _single_column_dmp_mf -> vmap over NUP=8 plumes:436   (vmap #2)
        plume_scan:357 -> lax.scan over nz-2 levels:425     (WHILE #1, trip 42)
          step body:361-419 -> _condensation_edmf:171:381
            lax.fori_loop(0, 16):197                        (WHILE #2, trip 16)

This module lowers the SAME arithmetic as ONE ``pl.pallas_call``: grid
programs tile the flattened (column x plume) lane axis, each lane runs the
42-level ``lax.fori_loop`` sweep inside the kernel, and the condensation
fixed point is invoked VERBATIM (``mynn_edmf._condensation_edmf``) inside the
step body, so the hot 16-loop carries zero transcription drift.  The step
body itself is transcribed op-for-op from ``_dmp_scan``'s ``step`` with the
reference association order and RUNTIME scalar constants (see below).

Launch economy: 672 (+ ~42 scan-body) launches/step -> 1 kernel launch per
``dmp_mf_columns`` call.  >=10x contract target exceeded ~700x on the family.
The device-time floor becomes the fp64 plume arithmetic itself, which the
device bake-off must confirm (this CPU window cannot measure it).

Semantics contract (frozen here, gated in
``tests/v025/test_m2_pbl_batch_fused_edmf.py``)
===========================================================================
* fp64 arithmetic with the REFERENCE OPERATION ORDER of ``_dmp_scan.step``
  (worktree ``mynn_edmf.py``) and ``_condensation_edmf`` (reused verbatim).
* **Step-body float constants are RUNTIME arguments** (the packed ``S`` f64
  vector), never Python-float closure constants.  Precedent (M2 bake-off #1,
  ``fused_vertical_implicit.py``): closed constants let the compilation
  context reassociate the written association order (1-ulp divergence);
  runtime scalars cannot be constant-folded, so the written association
  survives.  ``_condensation_edmf`` is the reference's own code traced from
  the same Python source, so its constants are identical by construction.
* Pre-registered CPU gate (declared BEFORE the first run, FMA-contraction
  precedent ``proofs/v025/econ/...VIOLATION.json`` and the sibling M2 test):
  two different compilation contexts (XLA reference vs Pallas interpreter)
  make independent FMA-contraction decisions at mul->add sites (the
  ``_qsat_blend`` Horner chains are exposed).  Bitwise equality is REPORTED
  but not required; the GATE is
  ``max |fused - reference| <= 1e-12 * max(1, max |reference|)`` per output,
  with the >=50% bitwise-fraction tripwire of the sibling M2 test.  A real
  reordering bug shows at 1e-1..5e-3 there; ulp noise shows at <=5e-14.
* The carried plume ``alive`` flag is stored as f64 0/1 (identical truth
  table to the reference's bool; both compile contexts support the
  comparison form).
* Dead-state parity: the reference carries ``qc_p`` between levels but never
  reads it (condensation restarts from ``qc = 0`` every level,
  ``_condensation_edmf``:186).  The kernel carries it identically, so the
  output cannot diverge on any input class.
* Environment gate: production default ``GPUWRF_EDMF_FUSED_PLUME`` unset/0
  keeps the XLA reference path (byte-identical default).  Device callers must
  export ``JAX_PALLAS_USE_MOSAIC_GPU=false`` on this jax build so the Triton
  lowering is selected (same requirement as ``fused_vertical_implicit``).

Integration seam (applied here, gated default-OFF)
===========================================================================
``mynn_edmf.dmp_mf_columns`` dispatches to :func:`dmp_mf_columns_fused` when
``GPUWRF_EDMF_FUSED_PLUME=1`` (trace-time read, same pattern as
``_cond_niter``).  The reference path (setup -> vmap(plume_scan) -> assemble)
is byte-identical to the pre-split monolith (bitwise A/B proven on the split;
see the sprint WORKER_REPORT.md).
"""

from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl

__all__ = [
    "TX",
    "fused_plume_scan",
    "dmp_mf_columns_fused",
]

#: Lane-window width (power of two; the Triton lowering requires it).
#: One lane = one (column, plume) pair.
TX = 128

_SLOTS = (
    # step-body constants of mynn_edmf._dmp_scan.step, in source order of
    # first appearance (all values are the exact Python floats of the source)
    "c0_3", "c0_0005", "c0_33", "c0_9", "c0_0003", "c5e_6", "c1500", "c4000",
    "c0_15", "c0_2", "c250", "c2", "c1_25", "c200", "c3", "c0_3333",
    "c1", "c0", "grav",
)
(_C03, _C0005, _C033, _C09, _C0003, _C5E6, _C1500, _C4000,
 _C015, _C02, _C250, _C2, _C125, _C200, _C3, _C03333,
 _C1, _C0, _GRAV) = range(len(_SLOTS))
_N_SLOTS = len(_SLOTS)


def _scalar_stack() -> jax.Array:
    """Pack the step-body constants (exact source values) into one f64 vector."""

    import math

    from gpuwrf.physics import mynn_edmf as _edmf

    data = [
        0.3, 0.0005, 0.33, 0.9, 0.0003, 5.0e-6, 1500.0, 4000.0,
        0.15, 0.2, 250.0, 2.0, 1.25, 200.0, 3.0, 0.3333,
        1.0, 0.0, float(_edmf.GRAV),
    ]
    assert len(data) == _N_SLOTS and math.isclose(data[_GRAV], 9.81)
    return jnp.asarray(data, jnp.float64)


def _plume_rise_kernel(
    # per-lane plume inits (lanes_pad,)
    l_ref, upa0_ref, upw0_ref, upthl0_ref, upqt0_ref, upqc0_ref,
    upu0_ref, upv0_ref,
    # env profiles, (nz [or nz+1 for zw], B) — scalar-first indexing
    thl_ref, qt1_ref, thv_ref, p_ref, dz_ref, zw_ref, u_ref, v_ref,
    pblh_ref,
    # runtime constant stack
    s_ref,
    # emits: (lanes_pad, nz-2) each
    ea_out, ew_out, eqt_out, eqc_out, ethl_out, eu_out, ev_out,
    *, nz: int, nup: int,
):
    """``_dmp_scan``'s plume integration for one TX lane window.

    Lane i covers column ``col = (pid*TX + i) // nup`` and plume
    ``(pid*TX + i) % nup``.  The level sweep ``k = 1..nz-2`` is a
    ``lax.fori_loop`` (lowers to one device loop); each iteration transcribes
    ``_dmp_scan.step`` (mynn_edmf.py) op-for-op with runtime scalars and calls
    the reference ``_condensation_edmf`` verbatim.
    """

    pid = pl.program_id(0)
    lanes = pid * TX + jnp.arange(TX, dtype=jnp.int32)
    col = jnp.minimum(lanes // nup, thl_ref.shape[1] - 1)

    f64 = s_ref.dtype
    s = lambda i: s_ref[i]
    c0_3, c0_0005, c0_33, c0_9, c0_0003 = s(_C03), s(_C0005), s(_C033), s(_C09), s(_C0003)
    c5e6, c1500, c4000, c0_15, c0_2 = s(_C5E6), s(_C1500), s(_C4000), s(_C015), s(_C02)
    c250, c2, c1_25, c200, c3, c0_3333 = s(_C250), s(_C2), s(_C125), s(_C200), s(_C3), s(_C03333)
    c1, c0, grav = s(_C1), s(_C0), s(_GRAV)

    l = l_ref[lanes]
    area_p = upa0_ref[lanes]
    w_p = upw0_ref[lanes]
    thl_p = upthl0_ref[lanes]
    qt_p = upqt0_ref[lanes]
    qc_p = upqc0_ref[lanes]
    u_p = upu0_ref[lanes]
    v_p = upv0_ref[lanes]
    alive = jnp.ones((TX,), dtype=f64)
    pblh = pblh_ref[col]

    def step(k, carry):
        w_p, thl_p, qt_p, qc_p, u_p, v_p, area_p, alive = carry

        zw_k = zw_ref[k, col]
        zw_k1 = zw_ref[k + 1, col]
        zw_km1 = zw_ref[k - 1, col]
        dz_k = dz_ref[k, col]
        dz_k1 = dz_ref[k + 1, col]
        thl_k = thl_ref[k, col]
        qt_k = qt1_ref[k, col]
        thv_k = thv_ref[k, col]
        thv_k1 = thv_ref[k + 1, col]
        p_k = p_ref[k, col]
        p_k1 = p_ref[k + 1, col]
        u_k = u_ref[k, col]
        v_k = v_ref[k, col]

        # entrainment (mynn_edmf.py:364-371) -- association order verbatim
        wmin_e = c0_3 + l * c0_0005
        ent = c0_33 / (jnp.minimum(jnp.maximum(w_p, wmin_e), c0_9) * l)
        ent = jnp.maximum(ent, c0_0003)
        ent_cap = jnp.minimum(pblh + c1500, c4000)
        ent = jnp.where(zw_k >= ent_cap, ent + (zw_k - ent_cap) * c5e6, ent)
        ent = jnp.minimum(ent, c0_9 / (zw_k1 - zw_k))
        entexp = ent * (zw_k1 - zw_k)

        # interface-averaged env values (:373-380)
        ak = dz_k1 / (dz_k1 + dz_k)
        bk = dz_k / (dz_k1 + dz_k)
        qtn = qt_p * (c1 - entexp) + qt_k * entexp
        thln = thl_p * (c1 - entexp) + thl_k * entexp
        pk = p_k * ak + p_k1 * bk

        # condensation fixed point -- REFERENCE CODE, verbatim (:381 -> :171)
        thvn, qcn = _edmf_condensation(qtn, thln, pk, zw_k1)

        thvk = thv_k * ak + thv_k1 * bk
        buoy = grav * (thvn / thvk - c1)
        bcoeff = jnp.where(buoy > c0, c0_15, c0_2)

        # w update with symmetric accel limiter (:387-394)
        dzc = jnp.minimum(zw_k - zw_km1, c250)
        wterm = (-c2 * ent * w_p + bcoeff * buoy / jnp.maximum(w_p, c0_2)) * dzc
        wn = w_p + wterm
        lim = jnp.minimum(c1_25 * (zw_k - zw_km1) / c200, c2)
        wn = jnp.minimum(wn, w_p + lim)
        wn = jnp.maximum(wn, w_p - lim)
        wn = jnp.minimum(jnp.maximum(wn, c0), c3)

        # momentum entrainment (:396-397)
        un = u_p * (c1 - entexp * c0_3333) + u_k * entexp * c0_3333
        vn = v_p * (c1 - entexp * c0_3333) + v_k * entexp * c0_3333

        still = (alive > c0) & (wn > c0)          # (:399), bool truth table equal
        area_n = jnp.where(still, area_p, c0)      # (:401)

        # emits at level k (:404-408, :417-418)
        ea_out[k - 1, lanes] = jnp.where(still, area_p, c0)
        ew_out[k - 1, lanes] = jnp.where(still, wn, c0)
        eqt_out[k - 1, lanes] = jnp.where(still, qtn, c0)
        eqc_out[k - 1, lanes] = jnp.where(still, qcn, c0)
        ethl_out[k - 1, lanes] = jnp.where(still, thln, c0)
        eu_out[k - 1, lanes] = jnp.where(still, un, c0)
        ev_out[k - 1, lanes] = jnp.where(still, vn, c0)

        new_carry = (                                # (:410-416)
            jnp.where(still, wn, w_p),
            jnp.where(still, thln, thl_p),
            jnp.where(still, qtn, qt_p),
            jnp.where(still, qcn, qc_p),
            jnp.where(still, un, u_p),
            jnp.where(still, vn, v_p),
            area_n,
            still.astype(f64),
        )
        return new_carry

    carry0 = (w_p, thl_p, qt_p, qc_p, u_p, v_p, area_p, alive)  # (:359)
    lax.fori_loop(1, nz - 1, step, carry0)  # ks = arange(1, nz-1), emit rows k-1


def _edmf_condensation(qt, thl, p, zagl):
    """Verbatim re-export shim (kept indirection one line for provenance)."""

    from gpuwrf.physics import mynn_edmf as _edmf

    return _edmf._condensation_edmf(qt, thl, p, zagl)


def fused_plume_scan(
    l_per_plume, upa0, upw0, upthl0, upqt0, upqc0, upu0, upv0, *,
    thl, qt1, thv, p, dz, zw, u, v, pblh,
    interpret: bool = True,
):
    """Drop-in replacement for ``jax.vmap(plume_scan)(...)`` (mynn_edmf.py:436).

    Inputs: the eight per-plume surface arrays shaped ``(B, NUP)``; the env
    profiles shaped ``(B, nz)`` (``zw``: ``(B, nz+1)``); ``pblh`` ``(B,)``.
    Returns the seven scan emit arrays ``(ea, ew, eqt, eqc, ethl, eu, ev)``
    shaped ``(B, NUP, nz-2)`` -- identical to the reference vmap shapes.

    ``interpret=True`` (default) runs the Pallas reference interpreter
    (correct on any backend incl. CPU -- the gate path); the device bake-off
    runs ``interpret=False`` with ``JAX_PALLAS_USE_MOSAIC_GPU=false``.
    """

    l_per_plume = jnp.asarray(l_per_plume, jnp.float64)
    B, nup = int(l_per_plume.shape[0]), int(l_per_plume.shape[1])
    nz = int(thl.shape[-1])
    if nz < 2:
        raise ValueError(f"need nz >= 2, got {nz}")
    n_lanes = B * nup
    pad = (-n_lanes) % TX
    n_lanes_pad = n_lanes + pad

    def pad_tail(x):
        x = jnp.asarray(x, jnp.float64).reshape(-1)
        if pad == 0:
            return x
        return jnp.concatenate([x, jnp.repeat(x[-1:], pad)])

    lane_args = (
        pad_tail(l_per_plume), pad_tail(upa0), pad_tail(upw0), pad_tail(upthl0),
        pad_tail(upqt0), pad_tail(upqc0), pad_tail(upu0), pad_tail(upv0),
    )
    t = lambda x: jnp.asarray(x, jnp.float64).T  # (B, nz) -> (nz, B): [k, col]
    env_args = (
        t(thl), t(qt1), t(thv), t(p), t(dz), t(zw), t(u), t(v),
        jnp.asarray(pblh, jnp.float64),
    )
    s = _scalar_stack()

    out_shape = jax.ShapeDtypeStruct((nz - 2, n_lanes_pad), jnp.float64)
    ea, ew, eqt, eqc, ethl, eu, ev = pl.pallas_call(
        lambda *refs: _plume_rise_kernel(*refs, nz=nz, nup=nup),
        grid=(n_lanes_pad // TX,),
        out_shape=[out_shape] * 7,
        interpret=interpret,
    )(*lane_args, *env_args, s)

    cut = lambda a: a.T[:n_lanes].reshape(B, nup, nz - 2)
    return cut(ea), cut(ew), cut(eqt), cut(eqc), cut(ethl), cut(eu), cut(ev)


def dmp_mf_columns_fused(sqw, sqv, sqc, u, v, w, th, thl, thv, tk, qke,
                         p, exner, rho, dz, zw, ust, flt, fltv, flq, flqv,
                         pblh, ts, dx, xland, dt, psig_shcu=None):
    """Batched ``mynn_edmf.dmp_mf_columns`` with the fused plume kernel.

    Identical signature and output dict.  Setup and assembly reuse the
    reference ``_dmp_setup`` / ``_dmp_assemble`` verbatim under ``vmap``
    (they are bulk vector ops, a handful of launches); ONLY the 672-dispatch
    plume nest is replaced.  ``tk/exner/ust/flqv/dt/sqc/th`` stay
    accepted-and-ignored exactly like the reference config.
    """

    from gpuwrf.physics import mynn_edmf as _edmf
    import os

    if os.environ.get("GPUWRF_MYNN_FP32_PLUME", "0") == "1":
        from gpuwrf.kernels.phys_mynn_plume import dmp_mf_columns_native
        return dmp_mf_columns_native(
            sqw, sqv, sqc, u, v, w, th, thl, thv, tk, qke,
            p, exner, rho, dz, zw, ust, flt, fltv, flq, flqv,
            pblh, ts, dx, xland, dt, psig_shcu=psig_shcu,
            interpret=jax.default_backend() == "cpu",
        )

    del sqc, tk, qke, exner, ust, flqv, dt, th  # unused in this config (ref parity)
    B = int(thl.shape[0])
    nz = int(thl.shape[-1])
    if psig_shcu is None:
        psig_shcu = jnp.ones((B,))

    setup = jax.vmap(
        lambda *a: _edmf._dmp_setup(*a[:-1], dx=dx, psig_shcu=a[-1])
    )(sqw, sqv, u, v, w, thv, thl, p, dz, zw, rho, pblh, ts, xland,
      flt, fltv, flq, psig_shcu)

    ea, ew, eqt, eqc, ethl, eu, ev = fused_plume_scan(
        setup["l_per_plume"], setup["upa0"], setup["upw0"], setup["upthl0"],
        setup["upqt0"], setup["upqc0"], setup["upu0"], setup["upv0"],
        thl=thl, qt1=setup["qt1"], thv=thv, p=p, dz=dz, zw=zw, u=u, v=v,
        pblh=pblh, interpret=jax.default_backend() == "cpu",
    )

    # WRF first-level veto + full_up, batched (verbatim mirror of the
    # per-column tail of _dmp_scan, mynn_edmf.py:447-463; elementwise).
    first_level_survives = jnp.all(ew[:, :, 0] > 0.0, axis=1)
    active = setup["active"] & first_level_survives
    zeros_top = jnp.zeros((B, _edmf.NUP, 1), dtype=thl.dtype)
    surf = lambda a: a[:, :, None]
    UPW = jnp.concatenate([surf(setup["upw0"]), ew, zeros_top], axis=2)
    UPQT = jnp.concatenate([surf(setup["upqt0"]), eqt, zeros_top], axis=2)
    UPQC = jnp.concatenate([surf(setup["upqc0"]), eqc, zeros_top], axis=2)
    UPTHL = jnp.concatenate([surf(setup["upthl0"]), ethl, zeros_top], axis=2)
    UPU = jnp.concatenate([surf(setup["upu0"]), eu, zeros_top], axis=2)
    UPV = jnp.concatenate([surf(setup["upv0"]), ev, zeros_top], axis=2)
    upa_scan = jnp.broadcast_to(setup["upa0"][:, :, None], (B, _edmf.NUP, nz - 2)) \
        * (ew > 0)
    UPA = jnp.concatenate([surf(setup["upa0"]), upa_scan, zeros_top], axis=2)

    def one(s_one, rho_c, thv_c, dz_c, fltv_c, active_c,
            upa_c, upw_c, upqt_c, upqc_c, upthl_c, upu_c, upv_c):
        return _edmf._dmp_assemble(
            s_one, rho_c, thv_c, dz_c, fltv_c, active_c,
            upa_c, upw_c, upqt_c, upqc_c, upthl_c, upu_c, upv_c)

    return jax.vmap(one)(
        setup, rho, thv, dz, fltv, active, UPA, UPW, UPQT, UPQC, UPTHL, UPU, UPV)
