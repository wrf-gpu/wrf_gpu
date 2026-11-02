"""CW02: detection power of CW01 — inject a synthetic coast-only WS10 excess of +1 % into GPU v0.4.2 and report
what the CW01 statistics would show (coast_sea / coast_land bias and GPU−CPU / GPU−GPU ratio per block). No JAX."""
import json
import sys

import numpy as np

sys.path.insert(0, 'proofs/v46-coastwind/tools')
import cw01_spread as c  # noqa: E402

cz = np.load(f'{c.S}/twin0227_d03.npz')
cpu, g42, g40 = c.load(f'{c.S}/twin0227_d03.npz'), c.load(c.TE), c.load(f'{c.S}/f20gpu0227_d03.npz')
ms = c.masks(cz['s_LANDMASK'] > 0.5)
coast = ms['coast_sea'] | ms['coast_land']
out = {}
for k in (0.0, 0.01):
    g = {f: v * np.where(coast, 1.0 + k, 1.0)[None] for f, v in g42.items()}
    for mn in ('coast_sea', 'coast_land', 'open_sea'):
        a = c.per_lead(g, cpu, ms[mn], 73)['WS']
        gg = c.per_lead(g, g40, ms[mn], 25)['WS']
        for bn, (lo, hi) in c.BLOCKS.items():
            e = {'gc_bias': float(np.mean(a['bias'][lo:hi + 1])), 'gc_rmse': c.block(a['rmse'], lo, hi)}
            r = c.block(gg['rmse'], lo, hi)
            if r:
                e['gc_over_gg'] = e['gc_rmse'] / r
            out[f'k{k}/{mn}/{bn}'] = e
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for key, e in out.items():
    print(key, ' '.join(f'{a}={b:+.4f}' for a, b in e.items()))
