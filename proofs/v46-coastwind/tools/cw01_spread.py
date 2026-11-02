"""CW01: is the GPU−CPU coastal 10 m wind difference systematic (port) or realization spread (chaos)? (v46-coastwind)

usage: python cw01_spread.py OUT.json
Data (read-only, all already used by o5-coast; no ALISIOS reads):
  CPU  = original CPU-WRF 4.7.1 twin 0227 d03 (o5-coast twin0227_d03.npz, 73 hourly frames)
  G042 = GPU v0.4.2 (w43-teacher te_gpu d03_sfc.npz, 73 frames)
  G040 = GPU v0.4.0 (F20 raw cache f20gpu0227_d03.npz, 25 frames)
  C    = original CPU-WRF vs CPU ±1 full-theta f32 ULP twin, d03 U10/V10 domain RMSE per lead
         (docs/release/evidence/v033/w2/W2bR/metrics.json 'pair_rmse'; whole domain, no masks)
Per lead: RMSE and mean (bias) of U10, V10 (grid-relative, as the CPU file) and WS10 for GPU−CPU and GPU−GPU per mask
(coast = 5x5 dilation of the land/sea edge, as w43-coastwind/o5-coast A1). Chaos signature: GPU−GPU (two port versions)
≈ GPU−CPU on every mask, coast/open ratio identical, bias ≈ 0 with no consistent sign, and GPU−CPU ≈ CPU−CPU once
saturated. A port defect at the coast would give GPU−CPU > GPU−GPU on coast masks and/or a persistent signed bias.
No JAX."""
import json
import sys

import numpy as np
from scipy.ndimage import binary_dilation

S = '<DATA_ROOT>/wrf_gpu2/scratch/o5-coast'
TE = '<USER_HOME>/wrf_gpu2_lanes/w43-teacher/te_gpu/20260227_gpu/d03_sfc.npz'
W2 = 'docs/release/evidence/v033/w2/W2bR/metrics.json'
BLOCKS = {'t1_6': (1, 6), 't7_24': (7, 24), 't25_47': (25, 47), 't48_72': (48, 72)}


def load(path, te=False):
    z = np.load(path)
    return {'U10': z['U10'].astype(np.float64), 'V10': z['V10'].astype(np.float64)}


def masks(lm):
    k = np.ones((5, 5), bool)
    edge = binary_dilation(lm, k) & binary_dilation(~lm, k)
    return {'all': np.ones_like(lm), 'open_sea': ~lm & ~edge, 'coast_sea': edge & ~lm, 'coast_land': edge & lm,
            'interior_land': lm & ~edge}


def per_lead(a, b, m, nlead):
    out = {}
    for f in ('U10', 'V10', 'WS'):
        if f == 'WS':
            x = np.hypot(a['U10'][:nlead], a['V10'][:nlead]) - np.hypot(b['U10'][:nlead], b['V10'][:nlead])
        else:
            x = a[f][:nlead] - b[f][:nlead]
        xm = x[:, m]
        out[f] = {'rmse': np.sqrt((xm ** 2).mean(axis=1)).tolist(), 'bias': xm.mean(axis=1).tolist()}
    return out


def block(series, lo, hi):
    s = np.asarray(series)
    hi = min(hi, len(s) - 1)
    if lo > hi:
        return None
    return float(np.sqrt((s[lo:hi + 1] ** 2).mean()))  # pooled RMS over the block (equal cells per lead)


def main(out):
    cz = np.load(f'{S}/twin0227_d03.npz')
    cpu = load(f'{S}/twin0227_d03.npz')
    g42 = load(TE)
    g40 = load(f'{S}/f20gpu0227_d03.npz')
    lm = cz['s_LANDMASK'] > 0.5
    assert np.array_equal(lm, np.load(TE)['LANDMASK'] > 0.5)
    ms = masks(lm)
    w2 = json.load(open(W2))['domains']['d03']['fields']
    C = {f: [r['pair_rmse'] for r in w2[f]['by_lead']] for f in ('U10', 'V10')}
    W2X = {f: [r['rmse'] for r in w2[f]['by_lead']] for f in ('U10', 'V10')}  # v0.3.3-era GPU candidate vs CPU
    pairs = {'g042_cpu': (g42, cpu, 73), 'g040_cpu': (g40, cpu, 25), 'g042_g040': (g42, g40, 25)}
    res = {'note': __doc__, 'masks': {k: int(v.sum()) for k, v in ms.items()}, 'per_lead': {}, 'blocks': {}}
    for pn, (a, b, n) in pairs.items():
        res['per_lead'][pn] = {mn: per_lead(a, b, m, n) for mn, m in ms.items()}
    res['per_lead']['cpu_cpu_ulp_all'] = {'all': {f: {'rmse': C[f]} for f in C}}
    res['per_lead']['w2_gpu033_cpu_all'] = {'all': {f: {'rmse': W2X[f]} for f in W2X}}
    for pn, d in res['per_lead'].items():
        for mn, fd in d.items():
            for f, v in fd.items():
                for bn, (lo, hi) in BLOCKS.items():
                    r = block(v['rmse'], lo, hi)
                    if r is None:
                        continue
                    e = res['blocks'].setdefault(bn, {}).setdefault(mn, {}).setdefault(f, {})
                    e[pn] = {'rmse': r}
                    if 'bias' in v:
                        bb = np.asarray(v['bias'][lo:min(hi, len(v['bias']) - 1) + 1])
                        e[pn].update(bias=float(bb.mean()), frac_leads_pos=float((bb > 0).mean()))
    # summary ratios
    summ = {}
    for bn, md in res['blocks'].items():
        for mn, fd in md.items():
            for f, e in fd.items():
                s = {}
                if 'g042_cpu' in e and 'g042_g040' in e:
                    s['GC042_over_GG'] = e['g042_cpu']['rmse'] / e['g042_g040']['rmse']
                if 'g040_cpu' in e and 'g042_g040' in e:
                    s['GC040_over_GG'] = e['g040_cpu']['rmse'] / e['g042_g040']['rmse']
                if 'cpu_cpu_ulp_all' in e:
                    s['GC042_over_C'] = e['g042_cpu']['rmse'] / e['cpu_cpu_ulp_all']['rmse']
                    s['W2gpu_over_C'] = e['w2_gpu033_cpu_all']['rmse'] / e['cpu_cpu_ulp_all']['rmse']
                if s:
                    summ.setdefault(bn, {}).setdefault(mn, {})[f] = s
    for bn in res['blocks']:
        for f in ('U10', 'V10', 'WS'):
            for pn in ('g042_cpu', 'g040_cpu', 'g042_g040'):
                try:
                    cs = res['blocks'][bn]['coast_sea'][f][pn]['rmse'] / res['blocks'][bn]['open_sea'][f][pn]['rmse']
                    cl = res['blocks'][bn]['coast_land'][f][pn]['rmse'] / res['blocks'][bn]['interior_land'][f][pn]['rmse']
                except KeyError:
                    continue
                summ.setdefault(bn, {}).setdefault('coast_amplification', {}).setdefault(f, {})[pn] = {
                    'coast_sea/open_sea': cs, 'coast_land/interior_land': cl}
    res['summary'] = summ
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
