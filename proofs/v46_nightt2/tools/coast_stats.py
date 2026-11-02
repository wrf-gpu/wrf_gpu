"""GPU-CPU T2/TSK/GLW mean over coastal land (land cell with a sea cell within 1 grid cell) and interior land, per frame; night/day summary.
Usage: coast_stats.py CPU GPU DOM"""
import sys, json, numpy as np, netCDF4
from scipy.ndimage import binary_dilation
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'scripts/release_plots'))
from wrfout_pairs import pairs
cpu, gpu, dom = sys.argv[1:4]
acc = {}
for t, c, g in pairs(cpu, gpu, dom):
    dc, dg = netCDF4.Dataset(c), netCDF4.Dataset(g)
    land = np.asarray(dc['LANDMASK'][0]) > 0.5
    coast = land & binary_dilation(~land, iterations=1)
    inner = land & ~coast
    sw = float(np.asarray(dc['SWDOWN'][0])[land].mean())
    ph = 'night' if sw < 1 else 'day'
    for f in ('T2', 'TSK', 'GLW'):
        d = np.asarray(dg[f][0], float) - np.asarray(dc[f][0], float)
        for name, m in (('coast', coast), ('inner', inner)):
            acc.setdefault((ph, name, f), []).append(float(d[m].mean()))
out = {f'{k[0]}/{k[1]}/{k[2]}': {'n_frames': len(v), 'mean': round(float(np.mean(v)), 4), 'min': round(min(v), 4), 'max': round(max(v), 4)} for k, v in sorted(acc.items())}
[print(k, v) for k, v in out.items()]
