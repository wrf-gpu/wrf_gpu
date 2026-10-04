"""WRF REAL32 BouLac parcel searches in one Pallas program per lane tile.

Each lane owns a (column, source level) search; potential energy, depth and
crossing state remain in registers. Source: module_bl_mynnedmf.F2192-2338.
"""
from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl

TX = 128


def _kernel(zw, dz, qtke, theta, lb1, lb2, *, nz):
    lane = pl.program_id(0) * TX + jnp.arange(TX, dtype=jnp.int32)
    col = jnp.minimum(lane // nz, dz.shape[0] - 1)
    # WRF copies the top-level result from the next level down.
    src = jnp.minimum(lane % nz, nz - 2)
    c = lambda value: jnp.asarray(value, jnp.float32)
    zero, one, half = c(0), c(1), c(.5)
    beta = c(9.81) / c(300)
    ts = theta[col, src]
    energy = qtke[col, src]
    zsrc = zw[col, src]
    dlu0 = zw[col, nz - 1] + dz[col, nz - 1] - zsrc - dz[col, src] * half
    zeros = jnp.zeros((TX,), jnp.float32)

    def up_cond(state):
        return jnp.max(state[-1].astype(jnp.float32)) > zero

    def up_step(state):
        idx, pe, depth, distance, alive = state
        safe = jnp.minimum(idx, nz - 2)
        delta = dz[col, safe]
        t0 = theta[col, safe]
        t1 = theta[col, safe + 1]
        # Keep the two source updates separate, in their WRF order.
        pe_new = pe - beta * ts * delta
        pe_new = pe_new + beta * (t1 + t0) * delta * half
        depth_new = depth + delta
        cross = alive & (energy < pe_new) & (energy >= pe)
        slope = (t1 - t0) / delta
        q = beta * (t0 - ts)
        rad = jnp.maximum(q * q + c(2) * slope * beta * (energy - pe), zero)
        quadratic = (-q + jnp.sqrt(rad)) / jnp.where(slope != zero, slope, one) / beta
        linear = (energy - pe) / (beta * jnp.where(t0 != ts, t0 - ts, one))
        fraction = jnp.where(slope != zero, quadratic, jnp.where(t0 != ts, linear, zero))
        distance = jnp.where(cross, depth_new - delta + fraction, distance)
        return (idx + 1, jnp.where(alive, pe_new, pe),
                jnp.where(alive, depth_new, depth), distance,
                alive & ~cross & (idx + 1 < nz - 1))

    _, _, _, dlu, _ = lax.while_loop(up_cond, up_step,
                                   (src, zeros, zeros, dlu0, src < nz - 1))

    def down_cond(state):
        return jnp.max(state[-1].astype(jnp.float32)) > zero

    def down_step(state):
        idx, pe, depth, distance, alive = state
        safe = jnp.maximum(idx, 1)
        delta = dz[col, safe - 1]
        t0 = theta[col, safe]
        t1 = theta[col, safe - 1]
        pe_new = pe + beta * ts * delta
        pe_new = pe_new - beta * (t1 + t0) * delta * half
        depth_new = depth + delta
        cross = alive & (energy < pe_new) & (energy >= pe)
        slope = (t0 - t1) / delta
        q = beta * (t0 - ts)
        rad = jnp.maximum(q * q + c(2) * slope * beta * (energy - pe), zero)
        quadratic = (q + jnp.sqrt(rad)) / jnp.where(slope != zero, slope, one) / beta
        linear = (energy - pe) / (beta * jnp.where(t0 != ts, t0 - ts, one))
        fraction = jnp.where(slope != zero, quadratic, jnp.where(t0 != ts, linear, zero))
        distance = jnp.where(cross, depth_new - delta + fraction, distance)
        return (idx - 1, jnp.where(alive, pe_new, pe),
                jnp.where(alive, depth_new, depth), distance,
                alive & ~cross & (idx - 1 > 0))

    _, _, _, dld, _ = lax.while_loop(down_cond, down_step,
                                   (src, zeros, zeros, zsrc, src > 0))
    dld = jnp.minimum(dld, zw[col, src + 1])
    dlu = jnp.maximum(c(.1), dlu / (one + dlu / c(1500)))
    dld = jnp.maximum(c(.1), dld / (one + dld / c(1500)))
    lb1[lane] = jnp.minimum(dlu, dld)
    lb2[lane] = jnp.sqrt(dlu * dld)


def boulac_length_native(zw, dz, qtke, theta, *, interpret=None):
    """Same last-axis profile interface; outputs and arithmetic are float32."""
    if interpret is None:
        interpret = jax.default_backend() == 'cpu'
    zw, dz, qtke, theta = [jnp.asarray(value, jnp.float32) for value in (zw, dz, qtke, theta)]
    shape = dz.shape
    nz = shape[-1]
    if nz < 2:
        raise ValueError('BouLac needs at least two levels')
    dz, qtke, theta = [value.reshape(-1, nz) for value in (dz, qtke, theta)]
    # The retained dispatcher supplies zw(kts:kte). The top interface is
    # reconstructed from zw(kte)+dz(kte), as in the reference implementation.
    zw = zw.reshape(dz.shape[0], -1)
    count = dz.shape[0] * nz
    padded = ((count + TX - 1) // TX) * TX
    out = jax.ShapeDtypeStruct((padded,), jnp.float32)
    a, b = pl.pallas_call(lambda *refs: _kernel(*refs, nz=nz),
                          grid=(padded // TX,), out_shape=(out, out),
                          interpret=interpret)(zw, dz, qtke, theta)
    return a[:count].reshape(shape), b[:count].reshape(shape)
