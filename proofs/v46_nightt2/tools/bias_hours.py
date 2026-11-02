"""Land-mean GPU-CPU differences per valid hour for surface fields. Usage: bias_hours.py CPU_DIR GPU_DIR DOMAIN [fields...]"""
import sys, pathlib, json, numpy as np, netCDF4
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'scripts/release_plots'))
from wrfout_pairs import pairs
cpu, gpu, dom = sys.argv[1:4]
fields = sys.argv[4:] or ['T2','TSK','Q2','GLW','SWDOWN','HFX','LH','GRDFLX','PBLH','U10','V10','UST','EMISS','ALBEDO']
out = []
for t, c, g in pairs(cpu, gpu, dom):
    dc, dg = netCDF4.Dataset(c), netCDF4.Dataset(g)
    land = (dc['LANDMASK'][0] > 0.5) & (dc['XICE'][0] < 0.5 if 'XICE' in dc.variables else True)
    row = {'t': t.isoformat()[5:13]}
    for f in fields:
        if f not in dc.variables or f not in dg.variables: continue
        a = np.asarray(dc[f][0], float); b = np.asarray(dg[f][0], float)
        if a.ndim != 2: a, b = a[0], b[0]
        d = (b - a)[land]
        row[f] = round(float(d.mean()), 3)
    if 'U10' in dc.variables:
        ws = lambda d: np.hypot(np.asarray(d['U10'][0], float), np.asarray(d['V10'][0], float))
        row['WS10'] = round(float((ws(dg) - ws(dc))[land].mean()), 3)
    out.append(row)
keys = list(out[0].keys())
print(' '.join(f'{k:>7}' for k in keys))
for r in out: print(' '.join(f'{str(r.get(k,"")):>7}' for k in keys))
