#!/usr/bin/env python3
"""Adversarial point set for calcnfromq / smallvalues / radardd02 / calc_eff_radius (density-scaled units).

Every value is an exactly representable float32 (written with repr of its float64 image), so the
fp32 and fp64 Fortran builds and the JAX port all start from identical inputs.
Columns: an(il=1..18), dn, t77, dt  (one line per point).
"""
import numpy as np

NA = 18
rng = np.random.default_rng(20261010)
pts = []

def base(dn=1.0, th=300.0, t77=1.0):
    a = np.zeros(NA + 1)
    a[1] = th; a[2] = 0.01; a[9] = 4.0816326e8
    return a, dn, t77

def P(**kw):
    a, dn, t77 = base(kw.pop('dn', 1.0), kw.pop('th', 300.0), kw.pop('t77', 1.0))
    idx = dict(th=1, qv=2, qc=3, qr=4, qi=5, qs=6, qh=7, qhl=8, ccn=9, nc=10, nr=11, ni=12, ns=13,
               nh=14, nhl=15, vh=16, vhl=17, ccna=18)
    for k, v in kw.items():
        a[idx[k]] = v
    pts.append((a, dn, t77))

# ---- calcnfromq-targeted
P()
P(qc=1e-3, nc=0, ccna=1e8)
P(qc=3e-3, nc=0)
P(qc=1e-14, nc=1e6)
P(qc=5e-9, nc=0)
P(qc=5e-9, nc=1e5)
P(qi=2e-5, ni=0, qr=5e-4, nr=0, qs=3e-4, ns=0, dn=0.8)
P(qi=1e-14, ni=10, qr=1e-13, nr=1, qs=1e-14, ns=1)
P(qr=5e-9, nr=5e-9, qi=5e-9, ni=5e-9, qs=5e-9, ns=5e-9)
P(qr=5e-9, nr=0.5e-9, qs=5e-9, ns=0.5e-9)
P(qh=2e-3, nh=0, vh=0, qhl=3e-3, nhl=0, vhl=0, dn=1.1)
P(qh=2e-3, nh=0, vh=4e-6, qhl=3e-3, nhl=0, vhl=3e-6)
P(qh=1e-13, nh=1, vh=1e-16, qhl=1e-13, nhl=1, vhl=1e-16)
P(qh=5e-9, nh=0, qhl=5e-9, nhl=0)
P(qh=1.0001e-8, nh=0, vh=0, qhl=1.0001e-8, nhl=0, vhl=0, dn=0.3)
P(qc=1e-3, nc=1e-9)
P(qr=1e-3, nr=5e-9, qs=1e-3, ns=5e-9)
# ---- smallvalues-targeted (hail)
P(qhl=1e-3, nhl=10, vhl=1e-3 / 400)
P(qhl=1e-3, nhl=10, vhl=1e-3 / 1200)
P(qhl=1e-3, nhl=10, vhl=0)
P(qhl=1e-3, nhl=1e-3, vhl=1e-3 / 900)
P(qhl=1e-3, nhl=1e9, vhl=1e-3 / 900)
P(qhl=5e-4, nhl=0, vhl=5e-4 / 900)
P(qhl=1e-13, nhl=1, vhl=1e-16)
P(qhl=1e-8, nhl=1e-6, vhl=1e-8 / 900)
# graupel
P(qh=1e-3, nh=10, vh=1e-3 / 100)
P(qh=1e-3, nh=10, vh=1e-3 / 1000)
P(qh=1e-3, nh=10, vh=0)
P(qh=1e-3, nh=1e-3, vh=1e-3 / 500)
P(qh=1e-3, nh=1e9, vh=1e-3 / 500)
P(qh=5e-4, nh=0, vh=5e-4 / 500)
P(qh=1e-13, nh=1, vh=1e-16)
P(qh=1e-8, nh=1e-6, vh=1e-8 / 500)
P(qh=1e-6, nh=1e-3, vh=1e-6 / 300, dn=0.5)
# snow / rain / ice / cloud folds and CCN restore
P(qs=1e-14, ns=1, th=260.0)
P(qs=1e-14, ns=1, th=290.0)
P(qs=1e-4, ns=0, th=260.0)
P(qr=1e-13, nr=1, qi=1e-13, ni=1)
P(qr=1e-4, nr=0, qi=1e-4, ni=0)
P(qc=1e-14, nc=1e6, ccna=1e8)
P(qc=1e-14, nc=1e6, ccna=1e8, qi=1e-5, ni=1e3)
P(qc=1e-14, nc=1e6, ccna=0.5, ccn=-5.0)
P(qc=1e-3, nc=-3.0, ccna=1e8)
P(qc=1e-14, nc=3e8, ccna=1e8)
# ---- radardd02 / eff radius targeted
P(qr=1e-3, nr=1e3, qs=1e-3, ns=1e3, qi=1e-4, ni=1e5, qh=1e-3, nh=10, vh=0, qhl=1e-2, nhl=1, vhl=0)
P(qr=1e-3, nr=5e-4, qs=1e-3, ns=1e-8, qi=1e-4, ni=0.5, qh=1e-3, nh=1e-9, vh=1e-3 / 500)
P(qh=1e-3, nh=10, vh=1e-3 / 50, qhl=1e-2, nhl=1, vhl=1e-2 / 100)
P(qh=1e-3, nh=10, vh=1e-3 / 1500, qhl=1e-2, nhl=1e12, vhl=1e-2 / 900)
P(qh=1e-3, nh=1e12, vh=1e-3 / 500, qhl=1e-2, nhl=1e-6, vhl=1e-2 / 900)
P(qc=1e-3, nc=1e8, qi=1e-6, ni=1e6, qs=1e-6, ns=1e2)
P(qc=1e-3, nc=1e3, qi=1e-3, ni=1, qs=1e-2, ns=1e-3)
P(qc=-1e-6, nc=1e8, qi=1e-5, ni=-1e3, qs=-1e-5, ns=1e3)

# ---- random combinations
mass_choices = [0.0, 1e-14, 5e-13, 2e-12, 5e-9, 2e-8, 1e-6, 1e-4, 1e-3, 5e-3]
num_choices = [0.0, -1.0, 5e-10, 5e-9, 1e-6, 1e-2, 1.0, 1e3, 1e6, 1e9]
for _ in range(150):
    a, dn, t77 = base(float(rng.uniform(0.3, 1.3)), float(rng.uniform(240, 320)), float(rng.uniform(0.7, 1.0)))
    for il in range(3, 9):
        a[il] = rng.choice(mass_choices) * float(rng.uniform(0.5, 2.0))
    for il in range(10, 16):
        a[il] = rng.choice(num_choices) * float(rng.uniform(0.5, 2.0))
    a[9] = rng.choice([4.0816326e8, 0.0, -10.0, 1e7])
    a[18] = rng.choice([0.0, 0.5, 2.0, 1e7, 1e8])
    for il, lv in ((7, 16), (8, 17)):
        rho = rng.choice([0.0, 50.0, 170.0, 400.0, 600.0, 900.0, 1500.0])
        a[lv] = 0.0 if rho == 0.0 else a[il] * dn / rho
    pts.append((a, dn, t77))

with open('adv_in.txt', 'w') as fh:
    fh.write(f'{len(pts)}\n')
    for a, dn, t77 in pts:
        vals = [float(np.float32(x)) for x in list(a[1:]) + [dn, t77]]
        fh.write(' '.join(repr(v) for v in vals) + '\n')
print('points', len(pts))
