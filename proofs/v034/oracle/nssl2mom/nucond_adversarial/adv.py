import sys, json
sys.path.insert(0, '<USER_HOME>/src/wrf_gpu2_wt/o1-nssl/src')
sys.path.insert(0, '<USER_HOME>/wrf_gpu2_lanes/o1-nssl/nucond_dev')
import numpy as np
import direct
import oracle_io as o
from gpuwrf.physics.nssl2mom.indices import *

def make(seed, ncol, nz_sub):
    rng = np.random.default_rng(seed)
    an = np.zeros((19, ncol, nz_sub)); cols = {k: np.zeros((ncol, nz_sub)) for k in ('dn1', 't77', 'pn', 'wn')}
    for j in range(ncol):
        c = int(rng.integers(1, 15)); sp = o.load_case('fp64', c)
        k0 = int(rng.integers(0, 40 - nz_sub + 1))
        sl = slice(k0, k0 + nz_sub)
        a = o.stage_an(sp, 'S3')[:, sl].copy()
        for k in cols: cols[k][j] = o.stage_col(sp, 'S0', k)[sl]
        n = nz_sub
        a[LV] *= rng.uniform(0.6, 2.4, n) ** rng.choice([0, 1], n, p=[0.3, 0.7])
        a[LT] += rng.uniform(-3, 3, n) * rng.choice([0, 1], n)
        m = rng.random(n)
        a[LC] = np.where(m < 0.25, 0.0, np.where(m < 0.4, rng.uniform(0, 2e-13, n), a[LC] * 10 ** rng.uniform(-2, 1, n)))
        a[LC] = np.where((a[LC] == 0) & (rng.random(n) < 0.5), rng.uniform(1e-6, 2e-3, n), a[LC])
        m = rng.random(n)
        a[LNC] = np.where(m < 0.15, 0.0, np.where(m < 0.3, rng.uniform(0, 2, n),
                 np.where(m < 0.45, rng.uniform(0, 1e7, n), (a[LNC] + 1e8) * 10 ** rng.uniform(-3, 1.5, n))))
        a[LR] = np.where(rng.random(n) < 0.5, a[LR] * 10 ** rng.uniform(-1, 1, n), rng.uniform(0, 3e-3, n))
        a[LNR] = (a[LNR] + 1e3) * 10 ** rng.uniform(-5, 4, n)
        a[LCCNA] = rng.uniform(0, 1.6, n) * a[LCCN]
        cols['wn'][j] = rng.uniform(-6, 9, n) * rng.choice([0.0, 1.0], n, p=[0.1, 0.9])
        an[:, j] = a
    return an, cols

if __name__ == '__main__':
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from gpuwrf.physics.nssl2mom.constants import get_constants
    from gpuwrf.physics.nssl2mom.indices import Prec
    from gpuwrf.physics.nssl2mom.nucond import nucond
    seed, ncol, nz_sub = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    dts = [6.0, 18.0, 54.0, 60.0, 120.0]
    out = {}
    for mode in ('fp64', 'fp32'):
        R = np.float32 if mode == 'fp32' else np.float64
        C = get_constants(mode); P = Prec(mode)
        worst = 0.0; worst_at = None
        for i, dt in enumerate(dts):
            an, cols = make(seed + i, ncol, nz_sub)
            an = an.astype(R); cols = {k: v.astype(R) for k, v in cols.items()}
            fa, ft0, fss = direct.run(mode, an, cols['dn1'], cols['t77'], cols['pn'], cols['wn'], dt)
            ja, jt0, jss = nucond(jnp.asarray(an), jnp.asarray(cols['dn1']), jnp.asarray(cols['t77']),
                                  jnp.asarray(cols['pn']), jnp.asarray(cols['wn']), dt, C, P)
            ja = np.asarray(ja, np.float64)
            assert np.all(np.isfinite(fa)), 'fortran nonfinite'
            for il in range(1, 19):
                r = fa[il]; p = ja[il]
                scale = np.max(np.abs(r)) or 1.0
                rel = np.abs(p - r) / np.maximum(np.abs(r), 1e-12 * scale)
                if rel.max() > worst:
                    worst = rel.max(); ix = np.unravel_index(rel.argmax(), rel.shape); worst_at = (dt, il, ix, r[ix], p[ix])
            for nm, r, p in (('t0', ft0, jt0), ('ss', fss, jss)):
                p = np.asarray(p, np.float64)
                rel = np.abs(p - r) / np.maximum(np.abs(r), 1e-12 * np.max(np.abs(r)))
                if rel.max() > worst:
                    worst = rel.max(); ix = np.unravel_index(rel.argmax(), rel.shape); worst_at = (dt, nm, ix, r[ix], p[ix])
        print(mode, 'worst rel %.3e' % worst, worst_at)
