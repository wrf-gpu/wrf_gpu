"""Build the compact NUCOND adversarial dataset (pristine direct-call NUCOND, fp32 + fp64)."""
import sys, hashlib
sys.path.insert(0, '<USER_HOME>/src/wrf_gpu2_wt/o1-nssl/src')
sys.path.insert(0, '<USER_HOME>/wrf_gpu2_lanes/o1-nssl/nucond_dev')
import numpy as np
import direct
from adv import make
from targeted import make_targeted
IN_SP = [1, 2, 3, 4, 9, 10, 11, 18]
OUT_SP = [1, 2, 3, 4, 10, 11, 18]
sets = {}
an, cols = make(101, 12, 20); sets['rand_dt60'] = (an, cols, 60.0)
an, cols = make(202, 12, 20); sets['rand_dt18'] = (an, cols, 18.0)
an, cols = make(303, 12, 20); sets['rand_dt120'] = (an, cols, 120.0)
an, cols = make_targeted(); sel = list(range(0, 24, 2))
sets['targeted_dt18'] = (an[:, sel], {k: v[sel] for k, v in cols.items()}, 18.0)
out = {}
for name, (an, cols, dt) in sets.items():
    for mode in ('fp32', 'fp64'):
        R = np.float32 if mode == 'fp32' else np.float64
        a = an.astype(R); c = {k: v.astype(R) for k, v in cols.items()}
        fa, ft0, fss = direct.run(mode, a, c['dn1'], c['t77'], c['pn'], c['wn'], dt)
        assert np.all(np.isfinite(fa)) and np.all(np.isfinite(fss))
        p = f'{name}/{mode}/'
        out[p + 'an_in'] = a[IN_SP].astype(np.float64)
        for k in ('dn1', 't77', 'pn', 'wn'):
            out[p + k] = c[k].astype(np.float64)
        out[p + 'an_out'] = fa[OUT_SP]
        out[p + 't0'] = ft0; out[p + 'ssat'] = fss
        out[p + 'dt'] = np.array(dt)
out['in_species'] = np.array(IN_SP); out['out_species'] = np.array(OUT_SP)
dst = sys.argv[1]
np.savez_compressed(dst, **out)
print(dst, hashlib.sha256(open(dst, 'rb').read()).hexdigest())
