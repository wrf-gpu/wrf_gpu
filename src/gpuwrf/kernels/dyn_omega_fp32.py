"""WRF calc_ww_cp stage omega (specified) as one REAL column kernel.

Same operands as dynamics/flux_advection.stage_omega_specified (edge-padded face
masses, coupled ru/rv, map-factor divergence). The column sum dmdt and the
recurrence ww(k+1) = ww(k) - dnw(k)*c1h(k)*dmdt - divv(k) run sequentially in k,
as module_big_step_utilities_em.F calc_ww_cp writes them, instead of an XLA
reduce + associative cumsum; rom is written row-major (no layout transpose).
Flag GPUWRF_DYN_GLUE_FUSED (omega).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

F = jnp.float32


def _omega_kernel(u, v, mu, c1h, c2h, dnw, msfuy, msfvx, msftx, out, *, nz, ny, nx, rdx, rdy, block):
    flat = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    j, i = flat // nx, flat % nx
    valid = flat < ny * nx
    m2 = lambda ref, jj, ii: pt.load(ref.at[jj, ii], mask=valid, other=F(0))
    mu_at = lambda jj, ii: m2(mu, jnp.clip(jj, 0, ny - 1), jnp.clip(ii, 0, nx - 1))
    # Face masses (stage_omega_specified): faces 0..n-1 average with the edge-padded left/lower
    # neighbour; the last face (n) carries the edge cell value itself.
    muu_w = F(.5) * (mu_at(j, i) + mu_at(j, i - 1))
    muu_e = jnp.where(i + 1 == nx, mu_at(j, i), F(.5) * (mu_at(j, i + 1) + mu_at(j, i)))
    muv_s = F(.5) * (mu_at(j, i) + mu_at(j - 1, i))
    muv_n = jnp.where(j + 1 == ny, mu_at(j, i), F(.5) * (mu_at(j + 1, i) + mu_at(j, i)))
    my_w, my_e = m2(msfuy, j, i), m2(msfuy, j, i + 1)
    mx_s, mx_n = m2(msfvx, j, i), m2(msfvx, j + 1, i)
    mt = m2(msftx, j, i)
    vec = lambda ref, k: pt.load(ref.at[k])

    def divv(k):
        c1, c2 = vec(c1h, k), vec(c2h, k)
        load = lambda ref, kk, jj, ii: pt.load(ref.at[kk, jj, ii], mask=valid, other=F(0))
        ru_w = (c1 * muu_w + c2) * load(u, k, j, i) / my_w
        ru_e = (c1 * muu_e + c2) * load(u, k, j, i + 1) / my_e
        rv_s = (c1 * muv_s + c2) * load(v, k, j, i) / mx_s
        rv_n = (c1 * muv_n + c2) * load(v, k, j + 1, i) / mx_n
        return mt * vec(dnw, k) * (F(rdx) * (ru_e - ru_w) + F(rdy) * (rv_n - rv_s))

    dmdt = jax.lax.fori_loop(0, nz, lambda k, acc: acc + divv(k), jnp.zeros_like(mt))
    pt.store(out.at[0, j, i], jnp.zeros_like(mt), mask=valid)

    def step(k, ww):
        ww = ww + (-(vec(dnw, k) * vec(c1h, k) * dmdt) - divv(k))
        pt.store(out.at[k + 1, j, i], jnp.where(k + 1 < nz, ww, F(0)), mask=valid)
        return ww
    jax.lax.fori_loop(0, nz, step, jnp.zeros_like(mt))


def stage_omega_fp32(u, v, mu_total, c1h, c2h, dnw, msfuy, msfvx, msftx, rdx, rdy, *, interpret=False, block=128, warps=4):
    nz, ny, nxs = u.shape
    nx = nxs - 1
    arrays = (u, v, mu_total, c1h, c2h, dnw, msfuy, msfvx, msftx)
    if any(a.dtype != jnp.float32 for a in arrays):
        raise TypeError("stage_omega_fp32 requires WRF REAL fp32 inputs")
    from gpuwrf.kernels.dyn_real_fp32 import pin_rows
    kernel = partial(_omega_kernel, nz=nz, ny=ny, nx=nx, rdx=rdx, rdy=rdy, block=block)
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((nz + 1, ny, nx), jnp.float32),
                          grid=((ny * nx + block - 1) // block,), name="b_diff_stage_omega_fp32", interpret=interpret,
                          compiler_params=pt.CompilerParams(num_warps=warps))(*pin_rows(arrays))
