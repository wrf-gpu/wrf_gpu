"""WRF REAL hydrostatic profiles in one column sweep (BP62/BP63 recurrence).

One lane carries the interface pressure down the column. Direct level-major
loads and stores avoid the unrolled XLA recurrence and its layout fusions.
No input packing or intermediate full-column moisture sum is allocated.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

from gpuwrf.kernels.dyn_rk_fp32 import _rn

BLOCK = 128


def _sweep(mut, qv, qc, qr, qi, qs, qg, c1h, c2h, dnw, p_top, faces, p_hyd,
           *, nz, ny, nx, interpret):
    lanes = pl.program_id(0) * BLOCK + jnp.arange(BLOCK, dtype=jnp.int32)
    live = lanes < ny * nx
    j, i = lanes // nx, lanes % nx
    rn = partial(_rn, interpret=interpret)
    one, half, zero = jnp.float32(1), jnp.float32(.5), jnp.float32(0)
    m = pt.load(mut.at[j, i], mask=live, other=zero)
    face = jnp.broadcast_to(p_top[()], (BLOCK,))
    pt.store(faces.at[nz, j, i], face, mask=live)
    for k in range(nz - 1, -1, -1):
        load = lambda ref: pt.load(ref.at[k, j, i], mask=live, other=zero)
        qtot = rn("add", rn("add", rn("add", rn("add", rn("add",
            load(qv), load(qc)), load(qr)), load(qi)), load(qs)), load(qg))
        mass = rn("add", rn("mul", jnp.broadcast_to(c1h[k], (BLOCK,)), m),
                  jnp.broadcast_to(c2h[k], (BLOCK,)))
        dp = rn("mul", rn("mul", rn("add", jnp.broadcast_to(one, (BLOCK,)), qtot), mass),
                jnp.broadcast_to(dnw[k], (BLOCK,)))
        below = rn("sub", face, dp)
        pt.store(faces.at[k, j, i], below, mask=live)
        average = rn("mul", jnp.broadcast_to(half, (BLOCK,)), rn("add", below, face))
        pt.store(p_hyd.at[k, j, i], average, mask=live)
        face = below


def hydrostatic_profiles(mut, moist, c1h, c2h, dnw, p_top, *, interpret=False):
    """Return REAL mass/interface profiles; caller owns storage-dtype conversion."""
    nz, ny, nx = moist[0].shape
    f32 = lambda value: jnp.asarray(value, jnp.float32)
    faces, p_hyd = pl.pallas_call(
        partial(_sweep, nz=nz, ny=ny, nx=nx, interpret=bool(interpret)),
        out_shape=(jax.ShapeDtypeStruct((nz + 1, ny, nx), jnp.float32),
                   jax.ShapeDtypeStruct((nz, ny, nx), jnp.float32)),
        grid=((-(-(ny * nx) // BLOCK)),), interpret=bool(interpret),
        compiler_params=pt.CompilerParams(num_warps=4),
        name="phy_prep_hydrostatic_real",
    )(f32(mut), *(f32(q) for q in moist), f32(c1h), f32(c2h), f32(dnw),
      f32(p_top).reshape(()))
    return p_hyd, faces
