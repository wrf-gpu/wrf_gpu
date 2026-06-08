"""JIT/vmap-batched Janjic Eta surface layer (``sf_sfclay_physics=2``).

This is the OPERATIONAL batched entrypoint for the Janjic Eta surface layer. The
single-column kernel ``physics.sfclay_janjic.myjsfc_column`` is ALREADY fully
``jax.jit``/``jax.vmap``-traceable (the SFCDIF Monin-Obukhov iteration is a
``jax.lax.scan``; the PSIM/PSIH integral functions are static device-table
lookups; everything else is pure ``jnp``). So unlike the MYJ PBL (whose host-
NumPy reference needed a traceable rewrite in ``physics.bl_myj``), the surface
layer only needs a ``jax.vmap`` wrapper over the ``(ncol, nz)`` grid columns --
no re-derivation. ``myjsfc_columns`` is the batched twin of ``myjsfc_column``.

Pairing: ``sf_sfclay_physics=2`` MUST run with ``bl_pbl_physics=2`` (MYJ PBL).
The surface layer runs FIRST in the WRF call chain and produces the exchange
coefficients / fluxes (USTAR, AKHS, AKMS, THZ0, QZ0, QSFC, CHKLOWQ, ELFLX) that
the paired MYJ PBL consumes; the operational scan + dispatcher enforce the pair.

fp64 throughout for savepoint parity; allocation-free; no host transfer.
"""

from __future__ import annotations

import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from gpuwrf.physics.sfclay_janjic import myjsfc_column


def myjsfc_columns(
    u, v, temperature, theta, qv, qc, p_mid, dz, q2,
    *, tsk, xland, z0base, psfc, znt, ustar, mavail,
    qsfc, thz0, qz0, uz0=0.0, vz0=0.0, pblh=1000.0, dt=60.0,
) -> dict:
    """``jax.vmap``-batched Janjic Eta surface layer over ``(ncol, nz)`` columns.

    Profile args are ``(ncol, nz)`` bottom-up (index 0 = lowest model layer).
    Surface scalars are ``(ncol,)`` (or broadcastable scalars). ``q2`` is TKE
    (m^2/s^2); MYJSFC converts to ``2*q2`` internally. Returns the batched SFCDIF
    output dict (each leaf ``(ncol,)``), identical keys to ``myjsfc_column``.
    """

    ncol = u.shape[0]
    f1 = lambda a: jnp.broadcast_to(jnp.asarray(a, jnp.float64).reshape(-1), (ncol,))

    def _one(u_c, v_c, t_c, th_c, qv_c, qc_c, pmid_c, dz_c, q2_c,
             tsk_c, xland_c, z0base_c, psfc_c, znt_c, ust_c, mav_c,
             qsfc_c, thz0_c, qz0_c, uz0_c, vz0_c, pblh_c):
        return myjsfc_column(
            u=u_c, v=v_c, temperature=t_c, theta=th_c, qv=qv_c, qc=qc_c,
            p_mid=pmid_c, dz=dz_c, q2=q2_c, tsk=tsk_c, xland=xland_c,
            z0base=z0base_c, psfc=psfc_c, znt=znt_c, ustar=ust_c, mavail=mav_c,
            dt=dt, qsfc=qsfc_c, thz0=thz0_c, qz0=qz0_c, uz0=uz0_c, vz0=vz0_c,
            pblh=pblh_c,
        )

    return jax.vmap(_one)(
        jnp.asarray(u, jnp.float64), jnp.asarray(v, jnp.float64),
        jnp.asarray(temperature, jnp.float64), jnp.asarray(theta, jnp.float64),
        jnp.asarray(qv, jnp.float64), jnp.asarray(qc, jnp.float64),
        jnp.asarray(p_mid, jnp.float64), jnp.asarray(dz, jnp.float64),
        jnp.asarray(q2, jnp.float64),
        f1(tsk), f1(xland), f1(z0base), f1(psfc), f1(znt), f1(ustar), f1(mavail),
        f1(qsfc), f1(thz0), f1(qz0), f1(uz0), f1(vz0), f1(pblh),
    )


__all__ = ["myjsfc_columns"]
