"""Native fp32 MYNN plume recurrence using one Pallas device program.

The retained WRF column oracle governs fidelity. This replaces only the
column-local plume sweep; setup reuses the reference equations and assembly
includes the WRF area limiter and dry-plume diagnostic sign.
"""
from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl

__all__ = [
    "TX",
    "fused_plume_scan",
    "dmp_mf_columns_native",
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
    """Pack the step-body constants (exact source values) into one fp32 vector."""

    import math

    from gpuwrf.physics import mynn_edmf as _edmf

    data = [
        0.3, 0.0005, 0.33, 0.9, 0.0003, 5.0e-6, 1500.0, 4000.0,
        0.15, 0.2, 250.0, 2.0, 1.25, 200.0, 3.0, 0.3333,
        1.0, 0.0, float(_edmf.GRAV),
    ]
    assert len(data) == _N_SLOTS and math.isclose(data[_GRAV], 9.81)
    return jnp.asarray(data, jnp.float32)


def _consts(s_ref):
    """Step-body constants from the packed stack, keyed by their _SLOTS name."""

    return {name: s_ref[i] for i, name in enumerate(_SLOTS)}


def _plume_level(k, carry, l, pblh, col, env, c):
    """One level of ``_dmp_scan.step`` (mynn_edmf.py:364-418) for a lane vector.

    Returns ``(still, emits, new_carry)``; emits are the seven level-k scan
    outputs ``(ea, ew, eqt, eqc, ethl, eu, ev)`` already masked by ``still``.
    """

    w_p, thl_p, qt_p, qc_p, u_p, v_p, area_p, alive = carry[:8]
    thl_ref, qt1_ref, thv_ref, p_ref, dz_ref, zw_ref, u_ref, v_ref = env[:8]
    c0, c1 = c["c0"], c["c1"]

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
    wmin_e = c["c0_3"] + l * c["c0_0005"]
    ent = c["c0_33"] / (jnp.minimum(jnp.maximum(w_p, wmin_e), c["c0_9"]) * l)
    ent = jnp.maximum(ent, c["c0_0003"])
    ent_cap = jnp.minimum(pblh + c["c1500"], c["c4000"])
    ent = jnp.where(zw_k >= ent_cap, ent + (zw_k - ent_cap) * c["c5e_6"], ent)
    ent = jnp.minimum(ent, c["c0_9"] / (zw_k1 - zw_k))
    entexp = ent * (zw_k1 - zw_k)

    # interface-averaged env values (:373-380)
    ak = dz_k1 / (dz_k1 + dz_k)
    bk = dz_k / (dz_k1 + dz_k)
    qtn = qt_p * (c1 - entexp) + qt_k * entexp
    thln = thl_p * (c1 - entexp) + thl_k * entexp
    pk = p_k * ak + p_k1 * bk

    # condensation fixed point -- REFERENCE CODE, verbatim (:381 -> :171)
    from gpuwrf.physics.mynn_edmf import _condensation_wrf_kwargs
    thvn, qcn = _edmf_condensation(qtn, thln, pk, zw_k1,
                                  **_condensation_wrf_kwargs(qc_p))

    thvk = thv_k * ak + thv_k1 * bk
    buoy = c["grav"] * (thvn / thvk - c1)
    bcoeff = jnp.where(buoy > c0, c["c0_15"], c["c0_2"])

    # w update with symmetric accel limiter (:387-394)
    dzc = jnp.minimum(zw_k - zw_km1, c["c250"])
    wterm = (-c["c2"] * ent * w_p + bcoeff * buoy / jnp.maximum(w_p, c["c0_2"])) * dzc
    wn = w_p + wterm
    lim = jnp.minimum(c["c1_25"] * (zw_k - zw_km1) / c["c200"], c["c2"])
    wn = jnp.minimum(wn, w_p + lim)
    wn = jnp.maximum(wn, w_p - lim)
    wn = jnp.minimum(jnp.maximum(wn, c0), c["c3"])

    # momentum entrainment (:396-397)
    un = u_p * (c1 - entexp * c["c0_3333"]) + u_k * entexp * c["c0_3333"]
    vn = v_p * (c1 - entexp * c["c0_3333"]) + v_k * entexp * c["c0_3333"]

    still = (alive > c0) & (wn > c0)          # (:399), bool truth table equal
    area_n = jnp.where(still, area_p, c0)      # (:401)

    # emits at level k (:404-408, :417-418)
    emits = tuple(jnp.where(still, x, c0) for x in (area_p, wn, qtn, qcn, thln, un, vn))
    new_carry = (                                # (:410-416)
        jnp.where(still, wn, w_p),
        jnp.where(still, thln, thl_p),
        jnp.where(still, qtn, qt_p),
        jnp.where(still, qcn, qc_p),
        jnp.where(still, un, u_p),
        jnp.where(still, vn, v_p),
        area_n,
        still.astype(alive.dtype),
    )
    if len(carry) == 9:
        # Passive QNI follows the same linear entrainment, with no feedback
        # into plume thermodynamics (WRF :6173/:6311).
        ni_n = carry[8] * (c1 - entexp) + env[8][k, col] * entexp
        emits = emits + (jnp.where(still, ni_n, c0),)
        new_carry = new_carry + (jnp.where(still, ni_n, carry[8]),)
    return still, emits, new_carry


def _plume_rise_kernel(
    # per-lane plume inits (lanes_pad,)
    l_ref, upa0_ref, upw0_ref, upthl0_ref, upqt0_ref, upqc0_ref,
    upu0_ref, upv0_ref,
    # env profiles, (nz [or nz+1 for zw], B) — scalar-first indexing
    thl_ref, qt1_ref, thv_ref, p_ref, dz_ref, zw_ref, u_ref, v_ref,
    pblh_ref, active_ref,
    # runtime constant stack
    s_ref,
    # emits: (nz-2, lanes_pad) each
    ea_out, ew_out, eqt_out, eqc_out, ethl_out, eu_out, ev_out,
    *, nz: int, nup: int,
):
    """``_dmp_scan``'s plume integration for one TX lane window.

    Lane i covers column ``col = (pid*TX + i) // nup`` and plume
    ``(pid*TX + i) % nup``.  The level sweep ``k = 1..nz-2`` is a
    device-resident while loop with tile early exit; each iteration transcribes
    ``_dmp_scan.step`` (mynn_edmf.py) op-for-op with runtime scalars and calls
    the reference ``_condensation_edmf`` verbatim (:func:`_plume_level`).
    """

    pid = pl.program_id(0)
    lanes = pid * TX + jnp.arange(TX, dtype=jnp.int32)
    col = jnp.minimum(lanes // nup, thl_ref.shape[1] - 1)

    work_dtype = s_ref.dtype
    c = _consts(s_ref)
    env = (thl_ref, qt1_ref, thv_ref, p_ref, dz_ref, zw_ref, u_ref, v_ref)

    l = l_ref[lanes]
    carry0 = (upw0_ref[lanes], upthl0_ref[lanes], upqt0_ref[lanes], upqc0_ref[lanes],  # (:359)
              upu0_ref[lanes], upv0_ref[lanes], upa0_ref[lanes],
              active_ref[col].astype(work_dtype))
    pblh = pblh_ref[col]
    outs = (ea_out, ew_out, eqt_out, eqc_out, ethl_out, eu_out, ev_out)

    def clear(k, unused):
        for out in outs:
            out[k - 1, lanes] = jnp.zeros((TX,), dtype=work_dtype)
        return unused

    lax.fori_loop(1, nz - 1, clear, jnp.int32(0))

    def keep(state):
        k, carry = state
        return (k < nz - 1) & (jnp.max(carry[-1]) > c["c0"])

    def advance(state):
        k, carry = state
        _still, emits, new_carry = _plume_level(k, carry, l, pblh, col, env, c)
        for out, value in zip(outs, emits):
            out[k - 1, lanes] = value
        return k + 1, new_carry

    lax.while_loop(keep, advance, (jnp.int32(1), carry0))


#: Per-level plume sums folded into the kernel (BP58), in output order.
_SUM_NAMES = ("aw", "awqt", "awqc", "awthl", "awu", "awv", "a", "a_w", "aqc", "aqt", "athl")


def _plume_sums_kernel(
    l_ref, upa0_ref, upw0_ref, upthl0_ref, upqt0_ref, upqc0_ref, upu0_ref, upv0_ref,
    thl_ref, qt1_ref, thv_ref, p_ref, dz_ref, zw_ref, u_ref, v_ref,
    pblh_ref, active_ref, rhoz_ref, s_ref, *outs, nz: int, nup: int,
    ni_init_ref=None, ni_env_ref=None,
):
    """Same plume sweep as :func:`_plume_rise_kernel`, but each level emits the
    per-column NUP sums the DMP assembly needs (F90:6363-6491) instead of the
    seven per-plume arrays: ``(nz-2, B_pad)`` sums + the first-level veto flag.

    With UPA = upa0*(ew>0), UPW = ew and wgt = rhoz_dmp(K)*UPA*UPW (Kmask = 1 on
    kernel levels) the sums are wgt, wgt*{UPQT,UPQC,UPTHL,UPU,UPV}, UPA, UPA*UPW,
    UPA*{UPQC,UPQT,UPTHL} -- the products in the order of ``_assemble_native``.
    """

    n_sums = len(_SUM_NAMES) + (ni_env_ref is not None)
    sum_outs, first_out = outs[:n_sums], outs[n_sums]
    pid = pl.program_id(0)
    lanes = pid * TX + jnp.arange(TX, dtype=jnp.int32)
    col = jnp.minimum(lanes // nup, thl_ref.shape[1] - 1)
    groups = TX // nup
    gcols = pid * groups + jnp.arange(groups, dtype=jnp.int32)

    work_dtype = s_ref.dtype
    c = _consts(s_ref)
    env = (thl_ref, qt1_ref, thv_ref, p_ref, dz_ref, zw_ref, u_ref, v_ref)
    gsum = lambda x: jnp.sum(x.reshape(groups, nup), axis=1)  # noqa: E731

    l = l_ref[lanes]
    upa0 = upa0_ref[lanes]
    carry0 = (upw0_ref[lanes], upthl0_ref[lanes], upqt0_ref[lanes], upqc0_ref[lanes],
              upu0_ref[lanes], upv0_ref[lanes], upa0, active_ref[col].astype(work_dtype))
    if ni_env_ref is not None:
        env = env + (ni_env_ref,)
        carry0 = carry0 + (ni_init_ref[lanes],)
    pblh = pblh_ref[col]

    def clear(k, unused):
        for out in sum_outs:
            out[k - 1, gcols] = jnp.zeros((groups,), dtype=work_dtype)
        return unused

    lax.fori_loop(1, nz - 1, clear, jnp.int32(0))

    def keep(state):
        k, carry, _first = state
        return (k < nz - 1) & (jnp.max(carry[7]) > c["c0"])

    def advance(state):
        k, carry, first = state
        still, emits, new_carry = _plume_level(k, carry, l, pblh, col, env, c)
        _ea, ew, eqt, eqc, ethl, eu, ev = emits[:7]
        upa = upa0 * still.astype(work_dtype)
        upa_w = upa * ew
        wgt = rhoz_ref[k, col] * upa_w
        values = (wgt, wgt * eqt, wgt * eqc, wgt * ethl, wgt * eu, wgt * ev,
                  upa, upa_w, upa * eqc, upa * eqt, upa * ethl)
        if ni_env_ref is not None:
            values = values + (wgt * emits[7],)
        for out, value in zip(sum_outs, values):
            out[k - 1, gcols] = gsum(value)
        # WRF first-level veto: every plume of the column survives level 1.
        first = jnp.where(k == 1, gsum(still.astype(work_dtype)), first)
        return k + 1, new_carry, first

    _k, _carry, first = lax.while_loop(
        keep, advance, (jnp.int32(1), carry0, jnp.zeros((groups,), work_dtype)))
    first_out[gcols] = (first == jnp.asarray(nup, work_dtype)).astype(work_dtype)


def _edmf_condensation(qt, thl, p, zagl, **kwargs):
    """Verbatim re-export shim (kept indirection one line for provenance)."""

    from gpuwrf.physics import mynn_edmf as _edmf

    return _edmf._condensation_edmf(qt, thl, p, zagl, **kwargs)


def fused_plume_scan(
    l_per_plume, upa0, upw0, upthl0, upqt0, upqc0, upu0, upv0, *,
    thl, qt1, thv, p, dz, zw, u, v, pblh, active=None,
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

    l_per_plume = jnp.asarray(l_per_plume, jnp.float32)
    B, nup = int(l_per_plume.shape[0]), int(l_per_plume.shape[1])
    nz = int(thl.shape[-1])
    if nz < 2:
        raise ValueError(f"need nz >= 2, got {nz}")
    n_lanes = B * nup
    pad = (-n_lanes) % TX
    n_lanes_pad = n_lanes + pad

    def pad_tail(x):
        x = jnp.asarray(x, jnp.float32).reshape(-1)
        if pad == 0:
            return x
        return jnp.concatenate([x, jnp.repeat(x[-1:], pad)])

    lane_args = (
        pad_tail(l_per_plume), pad_tail(upa0), pad_tail(upw0), pad_tail(upthl0),
        pad_tail(upqt0), pad_tail(upqc0), pad_tail(upu0), pad_tail(upv0),
    )
    t = lambda x: jnp.asarray(x, jnp.float32).T  # (B, nz) -> (nz, B): [k, col]
    env_args = (
        t(thl), t(qt1), t(thv), t(p), t(dz), t(zw), t(u), t(v),
        jnp.asarray(pblh, jnp.float32),
        jnp.ones((B,), jnp.float32) if active is None else jnp.asarray(active, jnp.float32),
    )
    s = _scalar_stack()

    out_shape = jax.ShapeDtypeStruct((nz - 2, n_lanes_pad), jnp.float32)
    ea, ew, eqt, eqc, ethl, eu, ev = pl.pallas_call(
        lambda *refs: _plume_rise_kernel(*refs, nz=nz, nup=nup),
        grid=(n_lanes_pad // TX,),
        out_shape=[out_shape] * 7,
        interpret=interpret,
    )(*lane_args, *env_args, s)

    cut = lambda a: a.T[:n_lanes].reshape(B, nup, nz - 2)
    return cut(ea), cut(ew), cut(eqt), cut(eqc), cut(ethl), cut(eu), cut(ev)


def plume_sums_enabled() -> bool:
    """GPUWRF_MYNN_PLUME_SUMS=1: NUP sums inside the plume kernel (BP58), default off."""

    import os

    return os.environ.get("GPUWRF_MYNN_PLUME_SUMS", "0") == "1"


def fused_plume_sums(
    l_per_plume, upa0, upw0, upthl0, upqt0, upqc0, upu0, upv0, *,
    thl, qt1, thv, p, dz, zw, u, v, pblh, rhoz_dmp, active=None, interpret: bool = True, qni=None,
):
    """Plume sweep returning the per-level NUP sums instead of the plume arrays.

    Returns ``(sums, first_ok)``: ``sums[name]`` is ``(nz-2, B)`` level-major for
    kernel levels k = 1..nz-2 (``_SUM_NAMES``); ``first_ok`` ``(B,)`` is 1 where
    every plume of the column survives level 1. ``rhoz_dmp`` is ``(B, nz)``.
    """

    l_per_plume = jnp.asarray(l_per_plume, jnp.float32)
    B, nup = int(l_per_plume.shape[0]), int(l_per_plume.shape[1])
    nz = int(thl.shape[-1])
    if TX % nup:
        raise ValueError(f"TX={TX} must be a multiple of NUP={nup}")
    n_lanes = B * nup
    pad = (-n_lanes) % TX
    n_lanes_pad = n_lanes + pad
    b_pad = n_lanes_pad // nup

    def pad_tail(x):
        x = jnp.asarray(x, jnp.float32).reshape(-1)
        if pad == 0:
            return x
        return jnp.concatenate([x, jnp.repeat(x[-1:], pad)])

    lane_args = (
        pad_tail(l_per_plume), pad_tail(upa0), pad_tail(upw0), pad_tail(upthl0),
        pad_tail(upqt0), pad_tail(upqc0), pad_tail(upu0), pad_tail(upv0),
    )
    t = lambda x: jnp.asarray(x, jnp.float32).T  # (B, nz) -> (nz, B): [k, col]
    env_args = (
        t(thl), t(qt1), t(thv), t(p), t(dz), t(zw), t(u), t(v),
        jnp.asarray(pblh, jnp.float32),
        jnp.ones((B,), jnp.float32) if active is None else jnp.asarray(active, jnp.float32),
        t(rhoz_dmp),
    )
    sum_shape = jax.ShapeDtypeStruct((nz - 2, b_pad), jnp.float32)
    names = _SUM_NAMES
    if qni is None:
        inputs = (*lane_args, *env_args, _scalar_stack())
        kernel = lambda *refs: _plume_sums_kernel(*refs, nz=nz, nup=nup)
    else:
        ni0 = (qni[:,0]*dz[:,1]+qni[:,1]*dz[:,0])/(dz[:,0]+dz[:,1])
        ni0 = jnp.broadcast_to(ni0[:,None],(B,nup))
        inputs = (*lane_args,*env_args,pad_tail(ni0),t(qni),_scalar_stack())
        kernel = lambda *refs: _plume_sums_kernel(*refs[:19],refs[21],*refs[22:],
            nz=nz,nup=nup,ni_init_ref=refs[19],ni_env_ref=refs[20])
        names = names + ('awqni',)
    outs = pl.pallas_call(
        kernel,
        grid=(n_lanes_pad // TX,),
        out_shape=[sum_shape] * len(names) + [jax.ShapeDtypeStruct((b_pad,), jnp.float32)],
        interpret=interpret,
    )(*inputs)
    sums = {name: out[:, :B] for name, out in zip(names, outs[:-1])}
    return sums, outs[-1][:B] > 0.0


def _assemble_from_sums(s, rho, thv, dz, fltv, active, sums, qni=None):
    """Batched :func:`_assemble_native` fed with in-kernel NUP sums (same F90:6363-6491 ops).

    Level 0 (surface plume inits) is summed here; kernel levels come from the
    plume kernel; the top level is zero (Kmask / zero plume top). The limiter
    ``adjustment`` (one factor per column) scales the UPA-linear sums after the
    plume sum instead of before it (rounding-level difference only).
    """
    from gpuwrf.physics import mynn_edmf as _edmf

    nz = dz.shape[-1]
    B = dz.shape[0]
    psig_w = s["psig_w"]
    rho_l, thv_l, dz_l = rho.T, thv.T, dz.T
    rhoz_dmp = jnp.concatenate([s["rhoz_mid"].T, rho_l[-1:]], axis=0)   # (nz, B)
    upa0, upw0 = s["upa0"], s["upw0"]                                    # (B, NUP)
    upa_w0 = upa0 * upw0
    wgt0 = rhoz_dmp[0][:, None] * upa_w0
    level0 = dict(aw=wgt0, awqt=wgt0 * s["upqt0"], awqc=wgt0 * s["upqc0"], awthl=wgt0 * s["upthl0"],
                  awu=wgt0 * s["upu0"], awv=wgt0 * s["upv0"], a=upa0, a_w=upa_w0,
                  aqc=upa0 * s["upqc0"], aqt=upa0 * s["upqt0"], athl=upa0 * s["upthl0"])
    zero_top = jnp.zeros((1, B), jnp.float32)
    full = {name: jnp.concatenate([jnp.sum(level0[name], axis=-1)[None], sums[name], zero_top], axis=0)
            for name in _SUM_NAMES}                                       # (nz, B)
    if qni is not None:
        ni0=(qni[:,0]*dz[:,1]+qni[:,1]*dz[:,0])/(dz[:,0]+dz[:,1])
        full['awqni']=jnp.concatenate([jnp.sum(wgt0*ni0[:,None],axis=-1)[None],sums['awqni'],zero_top],axis=0)

    def to_iface(inner):  # WRF s_aw1(K+1): shift up one level, [0]=0
        return jnp.concatenate([jnp.zeros((1, B), inner.dtype), inner], axis=0)[: nz + 1]

    s_aw = to_iface(full["aw"] * psig_w)
    s_awqt = to_iface(full["awqt"] * psig_w)
    s_awqc = to_iface(full["awqc"] * psig_w)
    s_awthl = to_iface(full["awthl"] * psig_w)
    s_awu = to_iface(full["awu"] * psig_w)
    s_awv = to_iface(full["awv"] * psig_w)
    s_awqv = s_awqt - s_awqc

    dzi0 = 0.5 * (dz_l[0] + dz_l[1])
    flx1 = jnp.where(s_aw[1] != 0.0,
                     jnp.maximum(s_aw[1] * (thv_l[0] - thv_l[1]) / dzi0, 1.0e-6),
                     0.0)
    flt2 = jnp.maximum(fltv, 0.0)
    need = (flx1 > _edmf.FLUXPORTION * flt2 / dz_l[0]) & (flx1 > 0.0)
    adjustment = jnp.where(need,
                           jnp.maximum(0.01, _edmf.FLUXPORTION * flt2 / dz_l[0] / jnp.maximum(flx1, 1e-30)),
                           1.0)                           # (B,)
    keep = lambda x: jnp.where(active, x, 0.0)  # noqa: E731
    s_aw, s_awqv, s_awqt, s_awqc = (keep(x * adjustment) for x in (s_aw, s_awqv, s_awqt, s_awqc))
    s_awthl, s_awu, s_awv = (keep(x * adjustment) for x in (s_awthl, s_awu, s_awv))

    edmf_a_inner = keep(full["a"] * adjustment * psig_w)
    edmf_aw_inner = full["a_w"] * adjustment * psig_w
    maxmf = jnp.max(keep(edmf_aw_inner), axis=0)
    upa_sum = full["a"] * adjustment
    safe = jnp.maximum(upa_sum, 1e-30)
    has_a = upa_sum > 0.0
    edmf_qc_inner = keep(jnp.where(has_a, full["aqc"] * adjustment / safe, 0.0))
    edmf_qt_inner = keep(jnp.where(has_a, full["aqt"] * adjustment / safe, 0.0))
    edmf_thl_inner = keep(jnp.where(has_a, full["athl"] * adjustment / safe, 0.0))
    maxmf = jnp.where(active & (jnp.max(edmf_qc_inner, axis=0) < 1e-8), -maxmf, maxmf)

    result = {
        "s_aw": s_aw.T, "s_awqv": s_awqv.T, "s_awqt": s_awqt.T, "s_awqc": s_awqc.T,
        "s_awthl": s_awthl.T, "s_awu": s_awu.T, "s_awv": s_awv.T,
        "edmf_a": edmf_a_inner.T, "edmf_qc": edmf_qc_inner.T, "edmf_qt": edmf_qt_inner.T,
        "edmf_thl": edmf_thl_inner.T, "maxmf": maxmf,
        "maxwidth": s["maxwidth"],
        "ztop_plume": _edmf._plume_top_height(s, full["a_w"].T > 0.0),
        "active": active.astype(jnp.float32), "psig_w": psig_w,
    }
    if _edmf._wrf_plume_velocity_needed():
        result["edmf_w"] = keep(jnp.where(has_a, full["a_w"] * adjustment / safe, 0.0)).T
    if qni is not None:
        result['s_awqni']=keep(to_iface(full['awqni']*psig_w)*adjustment).T
    return result


def _assemble_native(s, rho, thv, dz, fltv, active, UPA, UPW, UPQT, UPQC, UPTHL,
                  UPU, UPV):
    """DMP_mf s_aw*/diagnostics assembly (F90:6363-6491).

    v0.25 M2 family-#2 (pblbatch): verbatim extraction of the former
    `_single_column_dmp_mf` post-scan section.
    """
    from gpuwrf.physics import mynn_edmf as _edmf

    nz = dz.shape[-1]
    psig_w = s["psig_w"]
    rhoz_mid = s["rhoz_mid"]
    # ---- assemble s_aw* (lines 6363-6382): s_aw1(k+1) += rhoz(k)*UPA(K)*UPW(K)*Psig_w
    # for K=kts..kte-1 (0-based 0..nz-2). rhoz_dmp(K) is the interface ABOVE level K.
    rhoz_dmp = jnp.concatenate([rhoz_mid, rho[-1:]])  # length nz; [K]=interface above level K
    Kmask = (jnp.arange(nz) <= nz - 2).astype(jnp.float32)[None, :]  # K=0..nz-2
    upa_w = UPA * UPW  # (NUP, nz)
    wgt = rhoz_dmp[None, :] * upa_w * Kmask
    s_aw_inner = jnp.sum(wgt, axis=0) * psig_w          # length nz; index K
    s_awqt_inner = jnp.sum(wgt * UPQT, axis=0) * psig_w
    s_awqc_inner = jnp.sum(wgt * UPQC, axis=0) * psig_w
    s_awthl_inner = jnp.sum(wgt * UPTHL, axis=0) * psig_w
    s_awu_inner = jnp.sum(wgt * UPU, axis=0) * psig_w
    s_awv_inner = jnp.sum(wgt * UPV, axis=0) * psig_w
    # WRF writes these to s_aw1(K+1): shift up by one -> length nz+1, [0]=0
    def to_iface(inner):
        return jnp.concatenate([jnp.zeros((1,)), inner])  # length nz+1, drops last
    s_aw = to_iface(s_aw_inner)[: nz + 1]
    s_awqt = to_iface(s_awqt_inner)[: nz + 1]
    s_awqc = to_iface(s_awqc_inner)[: nz + 1]
    s_awthl = to_iface(s_awthl_inner)[: nz + 1]
    s_awu = to_iface(s_awu_inner)[: nz + 1]
    s_awv = to_iface(s_awv_inner)[: nz + 1]
    s_awqv = s_awqt - s_awqc  # line 6380

    # ---- flux limiter (lines 6423-6461) ----
    dzi0 = 0.5 * (dz[0] + dz[1])
    flx1 = jnp.where(s_aw[1] != 0.0,
                     jnp.maximum(s_aw[1] * (thv[0] - thv[1]) / dzi0, 1.0e-6),
                     0.0)
    flt2 = jnp.maximum(fltv, 0.0)
    need = (flx1 > _edmf.FLUXPORTION * flt2 / dz[0]) & (flx1 > 0.0)
    adjustment = jnp.where(need,
                           jnp.maximum(0.01, _edmf.FLUXPORTION * flt2 / dz[0] / jnp.maximum(flx1, 1e-30)),
                           1.0)
    s_aw = s_aw * adjustment
    s_awqt = s_awqt * adjustment
    s_awqc = s_awqc * adjustment
    s_awqv = s_awqv * adjustment
    s_awthl = s_awthl * adjustment
    s_awu = s_awu * adjustment
    s_awv = s_awv * adjustment
    # WRF module_bl_mynnedmf.F:6460 scales plume area in the same
    # limiter branch as the fluxes. Inactive limiter has adjustment=1.
    UPA = UPA * adjustment
    upa_w = UPA * UPW

    # zero everything if not active
    zero1 = jnp.zeros((nz + 1,))
    s_aw = jnp.where(active, s_aw, zero1)
    s_awqv = jnp.where(active, s_awqv, zero1)
    s_awqt = jnp.where(active, s_awqt, zero1)
    s_awqc = jnp.where(active, s_awqc, zero1)
    s_awthl = jnp.where(active, s_awthl, zero1)
    s_awu = jnp.where(active, s_awu, zero1)
    s_awv = jnp.where(active, s_awv, zero1)

    # edmf_a / maxmf diagnostics (mean over plumes; lines 6470-6491)
    edmf_a_inner = jnp.sum(UPA, axis=0) * psig_w        # length nz
    edmf_aw_inner = jnp.sum(upa_w, axis=0) * psig_w
    edmf_a_inner = jnp.where(active, edmf_a_inner, 0.0)
    maxmf = jnp.max(jnp.where(active, edmf_aw_inner, 0.0))

    # mean in-plume properties (WRF lines 870-889): area-weighted plume means
    # edmf_qt1/edmf_thl1/edmf_qc1 = sum(UPA*UPX)/sum(UPA) where sum(UPA)>0; only
    # edmf_a is multiplied by Psig_w. Consumed by the DMP shallow-cu
    # cldfra/qc_bl overwrite (mynn_sgs_cloud.dmp_shallow_cu_overwrite).
    upa_sum = jnp.sum(UPA, axis=0)                      # length nz, pre-Psig
    safe = jnp.maximum(upa_sum, 1e-30)
    has_a = upa_sum > 0.0
    edmf_qc_inner = jnp.where(has_a, jnp.sum(UPA * UPQC, axis=0) / safe, 0.0)
    edmf_qt_inner = jnp.where(has_a, jnp.sum(UPA * UPQT, axis=0) / safe, 0.0)
    edmf_thl_inner = jnp.where(has_a, jnp.sum(UPA * UPTHL, axis=0) / safe, 0.0)
    edmf_qc_inner = jnp.where(active, edmf_qc_inner, 0.0)
    edmf_qt_inner = jnp.where(active, edmf_qt_inner, 0.0)
    edmf_thl_inner = jnp.where(active, edmf_thl_inner, 0.0)
    # WRF:6740-6742 reports dry-plume maximum mass flux as negative.
    maxmf = jnp.where(active & (jnp.max(edmf_qc_inner) < 1e-8), -maxmf, maxmf)

    result = {
        "s_aw": s_aw,
        "s_awqv": s_awqv,
        "s_awqt": s_awqt,
        "s_awqc": s_awqc,
        "s_awthl": s_awthl,
        "s_awu": s_awu,
        "s_awv": s_awv,
        "edmf_a": edmf_a_inner,
        "edmf_qc": edmf_qc_inner,
        "edmf_qt": edmf_qt_inner,
        "edmf_thl": edmf_thl_inner,
        "maxmf": maxmf,
        "maxwidth": s["maxwidth"],
        "ztop_plume": _edmf._plume_top_height(s, jnp.any(UPW > 0.0, axis=0)),
        "active": active.astype(jnp.float32),
        "psig_w": psig_w,
    }
    if _edmf._wrf_plume_velocity_needed():
        result["edmf_w"] = jnp.where(active & has_a, jnp.sum(upa_w, axis=0) / safe, 0.0)
    return result


def _dmp_mf_columns_impl(sqw, sqv, sqc, u, v, w, th, thl, thv, tk, qke,
                         p, exner, rho, dz, zw, ust, flt, fltv, flq, flqv,
                         pblh, ts, dx, xland, dt, psig_shcu=None, cloud_base=None, *, interpret=True, qni=None):
    """Batched ``mynn_edmf.dmp_mf_columns`` with the fused plume kernel.

    Identical signature and output dict.  Setup reuses the
    reference ``_dmp_setup``; native assembly adds WRF area-limiter scaling
    and dry-plume maxmf sign under ``vmap``
    (they are bulk vector ops, a handful of launches); ONLY the level/condensation dispatch
    plume nest is replaced.  ``tk/exner/ust/flqv/dt/sqc/th`` stay
    accepted-and-ignored exactly like the reference config.
    """

    from gpuwrf.physics import mynn_edmf as _edmf

    del sqc, tk, qke, exner, ust, flqv, dt, th  # unused in this config (ref parity)
    B = int(thl.shape[0])
    nz = int(thl.shape[-1])
    if psig_shcu is None:
        psig_shcu = jnp.ones((B,))

    if cloud_base is None:
        setup = jax.vmap(
            lambda *a: _edmf._dmp_setup(*a[:-1], dx=dx, psig_shcu=a[-1])
        )(sqw, sqv, u, v, w, thv, thl, p, dz, zw, rho, pblh, ts, xland,
          flt, fltv, flq, psig_shcu)
    else:
        setup = jax.vmap(
            lambda *a: _edmf._dmp_setup(*a[:-2], dx=dx,
                psig_shcu=a[-2], cloud_base=a[-1])
        )(sqw, sqv, u, v, w, thv, thl, p, dz, zw, rho, pblh, ts, xland,
          flt, fltv, flq, psig_shcu, cloud_base)

    if plume_sums_enabled() or qni is not None:
        rhoz_dmp = jnp.concatenate([setup["rhoz_mid"], rho[:, -1:]], axis=-1)   # (B, nz)
        sums, first_ok = fused_plume_sums(
            setup["l_per_plume"], setup["upa0"], setup["upw0"], setup["upthl0"],
            setup["upqt0"], setup["upqc0"], setup["upu0"], setup["upv0"],
            thl=thl, qt1=setup["qt1"], thv=thv, p=p, dz=dz, zw=zw, u=u, v=v,
            pblh=pblh, rhoz_dmp=rhoz_dmp, active=setup["active"], interpret=interpret,
            **({} if qni is None else {'qni':qni}),
        )
        return _assemble_from_sums(setup, rho, thv, dz, fltv, setup["active"] & first_ok, sums,
            **({} if qni is None else {'qni':qni}))

    ea, ew, eqt, eqc, ethl, eu, ev = fused_plume_scan(
        setup["l_per_plume"], setup["upa0"], setup["upw0"], setup["upthl0"],
        setup["upqt0"], setup["upqc0"], setup["upu0"], setup["upv0"],
        thl=thl, qt1=setup["qt1"], thv=thv, p=p, dz=dz, zw=zw, u=u, v=v,
        pblh=pblh, active=setup["active"], interpret=interpret,
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
        return _assemble_native(
            s_one, rho_c, thv_c, dz_c, fltv_c, active_c,
            upa_c, upw_c, upqt_c, upqc_c, upthl_c, upu_c, upv_c)

    return jax.vmap(one)(
        setup, rho, thv, dz, fltv, active, UPA, UPW, UPQT, UPQC, UPTHL, UPU, UPV)


def dmp_mf_columns_native(*args, interpret=True, **kwargs):
    """Same DMP interface, with a local fp32 tracing scope.

    Only this REAL-only MYNN subgraph uses this context; Thompson DOUBLE
    islands elsewhere in the model retain x64 support.
    """
    with jax.enable_x64(False):
        args = tuple(jnp.asarray(a, dtype=jnp.float32) for a in args)
        kwargs = {k: (None if v is None else jnp.asarray(v, dtype=jnp.float32))
                  for k,v in kwargs.items()}
        return _dmp_mf_columns_impl(*args, interpret=interpret, **kwargs)
