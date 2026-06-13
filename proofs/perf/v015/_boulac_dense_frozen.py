"""Frozen dense (B,nz,nz) _boulac_length (pre-O(nz), commit 0b2a7066).

Used ONLY by the v0.15 perf/VRAM A/B harness to measure the dense-vs-scan
delta on the production d01 shape; NOT a production import.
"""
from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.physics.mynn_constants import GTR, NL_BOULAC_LMAX
from gpuwrf.physics.mynn_pbl import _edge_heights, _MYNN_BOULAC_FP32


def _boulac_length(zw, dz, qtke, theta):
    """Vectorized WRF ``boulac_length`` (``module_bl_mynnedmf.F:2192-2338``).

    For each level ``iz`` it integrates the buoyant displacement a parcel with
    TKE ``qtke(iz)`` can travel up (``dlu``) and down (``dld``) before its TKE is
    consumed by potential energy, then returns ``lb1=min(dlu,dld)`` and
    ``lb2=sqrt(dlu*dld)``. The Fortran does this with two nested data-dependent
    while loops per ``iz``; here the inner search is unrolled into the dense
    (nz x nz) PE-accumulation matrices and the first TKE-crossing is selected
    with a cumulative-OR mask. ``beta = gtr`` is the buoyancy coefficient.

    Arrays are last-axis (``..., nz``); ``zw`` is the WRF ``zw(kts:kte)`` subset
    (length nz). WRF references ``zw(kte+1)`` (the top interface) and
    ``zw(iz+1)``; we reconstruct those from the cumulative layer depths so no
    extra interface array is threaded through.
    """

    # v0.15 optional fp32 island (default OFF): the dense (B, nz, nz) parcel
    # search below is bandwidth-bound; computing it in fp32 halves the transient
    # HBM traffic.  Cast inputs in, cast (lb1, lb2) back to caller dtype at the
    # return.  Engaged via GPUWRF_MYNN_BOULAC_FP32=1; proven by the frozen
    # field-tolerance gate, not assumed bit-identical.
    _out_dtype = jnp.result_type(zw, dz, qtke, theta)
    if _MYNN_BOULAC_FP32 and _out_dtype == jnp.float64:
        zw = zw.astype(jnp.float32)
        dz = dz.astype(jnp.float32)
        qtke = qtke.astype(jnp.float32)
        theta = theta.astype(jnp.float32)

    beta = GTR
    nz = dz.shape[-1]
    # Full interface heights zwf(kts:kte+1), length nz+1: zwf[k]=sum(dz[:k]).
    zwf = _edge_heights(dz)                 # (..., nz+1); zwf[...,0]=0
    zw_top = zwf[..., -1:]                   # zw(kte+1)
    zw_kp1 = zwf[..., 1:]                     # zw(k+1) for k=0..nz-1, length nz

    i_idx = jnp.arange(nz)                    # source level iz
    j_idx = jnp.arange(nz)                    # target level izz
    src = i_idx[:, None]                      # (nz_src, 1)
    tgt = j_idx[None, :]                      # (1, nz_tgt)

    theta_i = theta[..., :, None]             # theta(iz) broadcast over izz
    theta_j = theta[..., None, :]             # theta(izz)
    dz_j = dz[..., None, :]                   # dz(izz)
    # theta(izz+1): shift; top level uses theta(nz-1) (only used where j<=nz-2).
    theta_jp1 = jnp.concatenate((theta[..., 1:], theta[..., -1:]), axis=-1)[..., None, :]
    theta_jm1 = jnp.concatenate((theta[..., :1], theta[..., :-1]), axis=-1)[..., None, :]
    dz_jm1 = jnp.concatenate((dz[..., :1], dz[..., :-1]), axis=-1)[..., None, :]

    # ---------------- UPWARD search (dlu) ----------------
    # zup increment at level izz (valid for iz<=izz<=nz-2):
    #   d_zup = beta*(theta(izz+1)+theta(izz))*dz(izz)*0.5 - beta*theta(iz)*dz(izz)
    up_incr = (beta * (theta_jp1 + theta_j) * dz_j * 0.5
               - beta * theta_i * dz_j)
    up_valid = (tgt >= src) & (tgt <= nz - 2)
    up_incr = jnp.where(up_valid, up_incr, 0.0)
    zup = jnp.cumsum(up_incr, axis=-1)        # zup after processing level izz
    zup_inf = jnp.concatenate((jnp.zeros_like(zup[..., :1]), zup[..., :-1]), axis=-1)
    zzz_up = jnp.cumsum(jnp.where(up_valid, dz_j, 0.0), axis=-1)  # depth iz..izz

    qtke_i = qtke[..., :, None]
    bbb_up = jnp.where(jnp.abs(theta_jp1 - theta_j) > 0.0,
                       (theta_jp1 - theta_j) / dz_j, 0.0)
    rad_up = jnp.maximum((beta * (theta_j - theta_i)) ** 2
                         + 2.0 * bbb_up * beta * (qtke_i - zup_inf), 0.0)
    tl_up_b = jnp.where(bbb_up != 0.0,
                        (-beta * (theta_j - theta_i) + jnp.sqrt(rad_up))
                        / jnp.where(bbb_up != 0.0, bbb_up * beta, 1.0),
                        0.0)
    tl_up_lin = jnp.where(theta_j != theta_i,
                          (qtke_i - zup_inf) / (beta * jnp.where(theta_j != theta_i, theta_j - theta_i, 1.0)),
                          0.0)
    tl_up = jnp.where(bbb_up != 0.0, tl_up_b, tl_up_lin)
    dlu_cand = zzz_up - dz_j + tl_up
    # WRF crossing: qtke(iz) < zup .and. qtke(iz) >= zup_inf, scanning izz upward.
    up_cross = up_valid & (qtke_i < zup) & (qtke_i >= zup_inf)
    up_first = up_cross & (jnp.cumsum(up_cross.astype(jnp.int32), axis=-1) == 1)
    dlu_default = zw_top[..., None] - zw[..., :, None] - dz[..., :, None] * 0.5
    dlu = jnp.where(jnp.any(up_first, axis=-1, keepdims=True),
                    jnp.sum(jnp.where(up_first, dlu_cand, 0.0), axis=-1, keepdims=True),
                    dlu_default)[..., 0]
    # iz==kte (top) cannot integrate upward -> keeps default (handled by mask).
    dlu = jnp.where(i_idx < nz - 1, dlu, dlu_default[..., 0])

    # ---------------- DOWNWARD search (dld) ----------------
    # at level izz (valid for kts+1<=izz<=iz, scanning izz downward from iz):
    #   d_zdo = beta*theta(iz)*dz(izz-1) - beta*(theta(izz-1)+theta(izz))*dz(izz-1)*0.5
    do_incr = (beta * theta_i * dz_jm1
               - beta * (theta_jm1 + theta_j) * dz_jm1 * 0.5)
    do_valid = (tgt <= src) & (tgt >= 1)
    do_incr = jnp.where(do_valid, do_incr, 0.0)
    # cumulative scanning DOWNWARD (decreasing izz) -> reverse-cumsum from izz=iz.
    zdo = jnp.cumsum(do_incr[..., ::-1], axis=-1)[..., ::-1]
    zdo_sup = jnp.concatenate((zdo[..., 1:], jnp.zeros_like(zdo[..., :1])), axis=-1)
    zzz_do = jnp.cumsum(jnp.where(do_valid, dz_jm1, 0.0)[..., ::-1], axis=-1)[..., ::-1]

    bbb_do = jnp.where(jnp.abs(theta_j - theta_jm1) > 0.0,
                       (theta_j - theta_jm1) / dz_jm1, 0.0)
    rad_do = jnp.maximum((beta * (theta_j - theta_i)) ** 2
                         + 2.0 * bbb_do * beta * (qtke_i - zdo_sup), 0.0)
    tl_do_b = jnp.where(bbb_do != 0.0,
                        (beta * (theta_j - theta_i) + jnp.sqrt(rad_do))
                        / jnp.where(bbb_do != 0.0, bbb_do * beta, 1.0),
                        0.0)
    tl_do_lin = jnp.where(theta_j != theta_i,
                          (qtke_i - zdo_sup) / (beta * jnp.where(theta_j != theta_i, theta_j - theta_i, 1.0)),
                          0.0)
    tl_do = jnp.where(bbb_do != 0.0, tl_do_b, tl_do_lin)
    dld_cand = zzz_do - dz_jm1 + tl_do
    do_cross = do_valid & (qtke_i < zdo) & (qtke_i >= zdo_sup)
    # first crossing scanning DOWNWARD: rank by reversed cumulative count.
    do_first = do_cross & (jnp.cumsum(do_cross[..., ::-1].astype(jnp.int32), axis=-1)[..., ::-1] == 1)
    dld_default = zw[..., :, None]
    dld = jnp.where(jnp.any(do_first, axis=-1, keepdims=True),
                    jnp.sum(jnp.where(do_first, dld_cand, 0.0), axis=-1, keepdims=True),
                    dld_default)[..., 0]
    dld = jnp.where(i_idx > 0, dld, dld_default[..., 0])

    # dld(iz) = min(dld(iz), zw(iz+1)); soft Lmax limit on both.
    dld = jnp.minimum(dld, zw_kp1)
    dlu = jnp.maximum(0.1, dlu / (1.0 + dlu / NL_BOULAC_LMAX))
    dld = jnp.maximum(0.1, dld / (1.0 + dld / NL_BOULAC_LMAX))

    lb1 = jnp.minimum(dlu, dld)
    lb2 = jnp.sqrt(dlu * dld)
    # WRF copies the top level from kte-1.
    lb1 = jnp.concatenate((lb1[..., :-1], lb1[..., -2:-1]), axis=-1)
    lb2 = jnp.concatenate((lb2[..., :-1], lb2[..., -2:-1]), axis=-1)
    if _MYNN_BOULAC_FP32 and _out_dtype == jnp.float64:
        lb1 = lb1.astype(_out_dtype)
        lb2 = lb2.astype(_out_dtype)
    return lb1, lb2


