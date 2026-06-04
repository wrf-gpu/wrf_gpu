import sys, os
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "true")
sys.path.insert(0, 'src')
import json, numpy as np
import jax.numpy as jnp
from gpuwrf.physics import _gf_reference as R
from gpuwrf.physics import _gf_jax as J


def setup(case):
    d = json.load(open(f'proofs/v060/savepoints/gf_case_{case}.json'))
    c = d['columns']; s = d['scalars']; kx = int(s['KX'])
    def to1(a):
        x = np.zeros(kx + 1); x[1:] = np.asarray(a, dtype=np.float64); return x
    cols = {k: to1(c[k]) for k in ('T', 'QV', 'P', 'PI', 'DZ', 'RHO', 'U', 'V', 'W', 'RTHBLTEN', 'RQVBLTEN')}
    return cols, s, kx


def prep(case):
    cols, s, kx = setup(case)
    t = cols['T']; q = cols['QV']; po_pa = cols['P']; pi = cols['PI']; dz = cols['DZ']; rho = cols['RHO']
    us = cols['U']; vs = cols['V']; w = cols['W']; rthbl = cols['RTHBLTEN']; rqvbl = cols['RQVBLTEN']
    dt = float(s['DT']); dx = float(s['DX']); hfx = float(s['HFX']); qfx = float(s['QFX'])
    kpbl = int(s['KPBL']); xland = float(s['XLAND'])
    po = np.zeros(kx + 1); po[1:] = po_pa[1:] * 0.01; psur = po_pa[1] * 0.01; ter11 = 0.0
    zo = np.zeros(kx + 1); zo[1] = ter11 + 0.5 * dz[1]
    for k in range(2, kx + 1):
        zo[k] = zo[k - 1] + 0.5 * (dz[k - 1] + dz[k])
    tn = np.zeros(kx + 1); qo = np.zeros(kx + 1); dhdt = np.zeros(kx + 1)
    tshall = np.zeros(kx + 1); qshall = np.zeros(kx + 1)
    q2d = q.copy()
    for k in range(1, kx + 1):
        if q2d[k] < 1e-8:
            q2d[k] = 1e-8
    for k in range(1, kx + 1):
        tn[k] = t[k] + rthbl[k] * pi[k] * dt
        qo[k] = q2d[k] + rqvbl[k] * dt
        tshall[k] = t[k] + rthbl[k] * pi[k] * dt
        dhdt[k] = 1004.0 * rthbl[k] * pi[k] + 2.5e6 * rqvbl[k]
        qshall[k] = q2d[k] + rqvbl[k] * dt
        if tn[k] < 200:
            tn[k] = t[k]
        if qo[k] < 1e-8:
            qo[k] = 1e-8
    omeg = np.zeros(kx + 1)
    for k in range(1, kx + 1):
        omeg[k] = -9.81 * rho[k] * w[k]
    mconv = 0.0
    for k in range(1, kx):
        mconv += omeg[k] * (q2d[k + 1] - q2d[k]) / 9.81
    if mconv < 0:
        mconv = 0.0
    return dict(kx=kx, t=t, q2d=q2d, po=po, psur=psur, ter11=ter11, zo=zo, tn=tn,
                qo=qo, dhdt=dhdt, tshall=tshall, qshall=qshall, omeg=omeg,
                mconv=mconv, us=us, vs=vs, rho=rho, dt=dt, dx=dx, hfx=hfx,
                qfx=qfx, kpbl=kpbl, xland=xland, pi=pi)


def J1(a):
    return jnp.asarray(a, jnp.float64)


def check_deep(case):
    p = prep(case)
    kx = p['kx']
    ccn = 150.0
    dpr = R.cup_gf(1, 0, ccn, p['dt'], 0, p['kpbl'], p['dhdt'].copy(), p['xland'],
                   p['zo'].copy(), p['t'].copy(), p['q2d'].copy(), p['ter11'],
                   p['tn'].copy(), p['qo'].copy(), p['po'].copy(), p['psur'],
                   p['us'].copy(), p['vs'].copy(), p['rho'].copy(), p['hfx'],
                   p['qfx'], p['dx'], p['mconv'], p['omeg'].copy(), 0, kx)
    djr = J.cup_gf(1, 0, ccn, p['dt'], 0, p['kpbl'], J1(p['dhdt']), float(p['xland']),
                   J1(p['zo']), J1(p['t']), J1(p['q2d']), float(p['ter11']),
                   J1(p['tn']), J1(p['qo']), J1(p['po']), float(p['psur']),
                   J1(p['us']), J1(p['vs']), J1(p['rho']), float(p['hfx']),
                   float(p['qfx']), float(p['dx']), float(p['mconv']),
                   J1(p['omeg']), 0.0, kx)
    print(f'=== case {case} DEEP ===')
    print('ref: ierr=%d kbcon=%d ktop=%d k22=%d pre=%.6e xmb=%.6e' % (
        dpr['ierr'], dpr['kbcon'], dpr['ktop'], dpr['k22'], dpr['pre'], dpr['xmb_out']))
    print('jax: ierr=%d kbcon=%d ktop=%d k22=%d pre=%.6e xmb=%.6e' % (
        int(djr['ierr']), int(djr['kbcon']), int(djr['ktop']), int(djr['k22']),
        float(djr['pre']), float(djr['xmb_out'])))
    for fld in ['outt', 'outq', 'outqc']:
        a = np.array(dpr[fld]); b = np.array(djr[fld])
        sc = max(np.max(np.abs(a)), 1e-30)
        print('  %s: maxabs=%.3e maxrel=%.3e ref_max=%.3e' % (fld, np.max(np.abs(a - b)), np.max(np.abs(a - b)) / sc, np.max(np.abs(a))))


if __name__ == '__main__':
    cases = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4, 5]
    for cc in cases:
        check_deep(cc)


def check_shallow(case):
    p = prep(case)
    kx = p['kx']
    spr = R.cup_gf_sh(p['zo'].copy(), p['t'].copy(), p['q2d'].copy(), p['ter11'],
                      p['tshall'].copy(), p['qshall'].copy(), p['po'].copy(),
                      p['psur'], p['dhdt'].copy(), p['kpbl'], p['rho'].copy(),
                      p['hfx'], p['qfx'], p['xland'], 0, 258.0, p['dt'], kx)
    sjr = J.cup_gf_sh(J1(p['zo']), J1(p['t']), J1(p['q2d']), float(p['ter11']),
                      J1(p['tshall']), J1(p['qshall']), J1(p['po']), float(p['psur']),
                      J1(p['dhdt']), p['kpbl'], J1(p['rho']), float(p['hfx']),
                      float(p['qfx']), float(p['xland']), 0, 258.0, p['dt'], kx)
    print('=== case %d SHALLOW ===' % case)
    print('ref: ierr=%d kbcon=%d ktop=%d k22=%d pre=%.6e xmb=%.6e' % (
        spr['ierr'], spr['kbcon'], spr['ktop'], spr['k22'], spr['pre'], spr['xmb_out']))
    print('jax: ierr=%d kbcon=%d ktop=%d k22=%d pre=%.6e xmb=%.6e' % (
        int(sjr['ierr']), int(sjr['kbcon']), int(sjr['ktop']), int(sjr['k22']),
        float(sjr['pre']), float(sjr['xmb_out'])))
    for fld in ['outt', 'outq', 'outqc']:
        a = np.array(spr[fld]); b = np.array(sjr[fld])
        sc = max(np.max(np.abs(a)), 1e-30)
        print('  %s: maxabs=%.3e maxrel=%.3e' % (fld, np.max(np.abs(a - b)), np.max(np.abs(a - b)) / sc))
