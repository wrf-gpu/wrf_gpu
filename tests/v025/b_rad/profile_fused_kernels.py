"""One real band/tile of each fused transfer kernel for ncu roofline."""

import argparse
import os

import jax
import jax.numpy as jnp
import nvtx

from bench_radiation import load_columns
from gpuwrf.kernels import rad_lw_transfer, rad_sw_quadrature
from gpuwrf.physics import rrtmg_lw as lw, rrtmg_sw as sw
from gpuwrf.physics.rrtmg_tables import RRTMG_TABLES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=['lw', 'sw'], required=True)
    opts = p.parse_args()
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    assert jax.devices()[0].platform == 'gpu'
    sw_state, lw_state, _ = load_columns('d02', '12')
    def tile(state):
        n = state.p.shape[0]
        return jax.tree.map(lambda a: a[:1024] if getattr(a, 'ndim', 0) > 0 and a.shape[0] == n else a, state)
    if opts.family == 'lw':
        state, coef, secdiff, plank, plevel, psurf, cf, tc, _, _ = lw._lw_solver_base(
            tile(lw_state), RRTMG_TABLES, build_taumol=False)
        band = 2
        tau, frac = lw._lw_taumol_band(band, coef, lw._native_lw_tables(), RRTMG_TABLES)
        cloud = jnp.any(cf > .5, axis=(-1, -2))
        args = (state, tau, frac, cf[..., band, :], tc[..., band, :], secdiff[..., band],
                RRTMG_TABLES.lw_delwave[band] * jnp.pi * 1e4, RRTMG_TABLES.lw_gpoint_mask[band],
                plank[..., band], plevel[..., band], psurf[..., band], cloud, True)
        fn = jax.jit(rad_lw_transfer.lw_band_fluxes, static_argnums=12)
    else:
        state = tile(sw_state)
        data = sw.compute_rrtmg_sw_intermediates(state)
        # Intermediate reftra arrays retain top-down order and the surface.
        band = 2
        args = tuple(a[..., band:band + 1, :] for a in (
            data.spcvmc_zref, data.spcvmc_zrefd, data.spcvmc_ztra,
            data.spcvmc_ztrad, data.spcvmc_direct_trans))
        fn = jax.jit(rad_sw_quadrature.vertical_quadrature)
    exe = fn.lower(*args).compile()
    runtime_args = args[:-1] if opts.family == 'lw' else args
    jax.block_until_ready(exe(*runtime_args))
    with nvtx.annotate('BR_roof_' + opts.family):
        jax.block_until_ready(exe(*runtime_args))


if __name__ == '__main__':
    main()
