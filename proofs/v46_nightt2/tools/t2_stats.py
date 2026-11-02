"""Per-hour land T2 GPU-CPU mean/RMSE/p05/p95 + nighttime summary. Usage: t2_stats.py CPU GPU DOM"""
import sys, numpy as np, netCDF4
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'scripts/release_plots'))
from wrfout_pairs import pairs
cpu, gpu, dom = sys.argv[1:4]
night=[]
for t, c, g in pairs(cpu, gpu, dom):
    dc, dg = netCDF4.Dataset(c), netCDF4.Dataset(g)
    land = dc['LANDMASK'][0] > 0.5
    d = (np.asarray(dg['T2'][0], float) - np.asarray(dc['T2'][0], float))[land]
    sw = float(np.asarray(dc['SWDOWN'][0])[land].mean())
    if sw < 1: night.append(d.mean())
    print(f"{t:%m-%dT%H} n={d.size} mean={d.mean():+.3f} rmse={np.sqrt((d*d).mean()):.3f} p05={np.percentile(d,5):+.3f} p95={np.percentile(d,95):+.3f} cpuSW={sw:.0f}")
print('night frames', len(night), 'mean of means', np.mean(night) if night else None)
