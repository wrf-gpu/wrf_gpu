import sys
sys.path.insert(0, '<USER_HOME>/src/wrf_gpu2_wt/o1-nssl/src')
sys.path.insert(0, '<USER_HOME>/wrf_gpu2_lanes/o1-nssl/nucond_dev')
import numpy as np
import oracle_io as o
from gpuwrf.physics.nssl2mom.indices import *

def make_targeted():
    """Columns aimed at the NUCOND branches the 14 savepoint cases never reach."""
    cols_out = []
    ans = []
    rng = np.random.default_rng(7)
    for j in range(30):
        sp = o.load_case('fp64', [1, 4, 11, 14][j % 4])
        a = o.stage_an(sp, 'S3').copy()
        cols = {k: o.stage_col(sp, 'S0', k).copy() for k in ('dn1', 't77', 'pn', 'wn')}
        t0 = a[LT] * cols['t77']
        # qvs as in NUCOND (Bolton, iqvsopt=1)
        l = np.clip(np.trunc((t0 - 163.15) / 0.002 + 1.5), 1, 1000001)
        temq = 163.15 + (l - 1) * 0.002
        tq = np.exp(17.67 * (temq - 273.15) / (temq - 29.65))
        qvs = 0.622 * 611.2 * tq / (cols['pn'] - 611.2 * tq)
        if j < 12:   # few droplets + heavy rain: dtcon1 = 0.05 but huge rain relaxation -> halving -> EXIT
            a[LV] = qvs * (1.0 + rng.uniform(0.002, 0.05, 40))
            a[LC] = rng.uniform(2e-5, 3e-4, 40)
            a[LNC] = rng.uniform(1.0, 5e4, 40)
            a[LR] = rng.uniform(1e-3, 8e-3, 40)
            a[LNR] = rng.uniform(1e5, 5e7, 40)
        elif j >= 24:  # extreme droplet number (cx>1e6 branch keeps cx): first predictor undershoots -> EXIT (11066)
            a[LV] = qvs * (1.0 + rng.uniform(0.005, 0.3, 40))
            a[LC] = rng.uniform(1e-4, 2e-3, 40)
            a[LNC] = 10 ** rng.uniform(11, 15, 40)
        else:        # near-saturation: |ss1-1| <= 1e-5 -> delta = 0.1*dtp
            a[LV] = qvs * (1.0 + rng.uniform(1e-7, 9e-6, 40))
            a[LC] = rng.uniform(1e-6, 1e-3, 40)
            a[LNC] = rng.uniform(1e7, 5e8, 40)
            a[LR] = rng.uniform(0, 1e-3, 40)
            a[LNR] = rng.uniform(1e2, 1e5, 40)
        a[LCCNA] = rng.uniform(0, 1.2, 40) * a[LCCN]
        cols['wn'] = rng.uniform(-2, 6, 40)
        ans.append(a); cols_out.append(cols)
    an = np.stack(ans, axis=1)
    cols = {k: np.stack([c[k] for c in cols_out]) for k in cols_out[0]}
    return an, cols
