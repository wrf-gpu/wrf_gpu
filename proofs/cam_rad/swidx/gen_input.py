"""SWIDX input: the 210 CAM01 radctl SW operand columns + seeded synthetic overlap-stress columns.

Usage: python gen_input.py <cam01_fixture.npz> <out.bin> <mode 0|1> <columns.json>
Synthetic columns reuse real CAM01 day profiles (t, q1, p, o3, rel, rei) and replace cloud fields to exercise:
the totwgt = 0 maximum-overlap second pass, > nconfgmax qualifying configurations with tied weights (findvalue),
cld = 1 layers, mixed liquid/ice, very low sun.  pmxrgn/nmxrgn come from the port's cldovrlap (radcswmx input only).
"""
import json
import sys

import numpy as np

from gpuwrf.physics.ra_cam_common import cldovrlap

fx, out, mode, cols_json = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = np.load(fx)
nz = d["r8s_q1"].shape[1]
keys = ["q1", "cld", "pmid", "pint", "t", "cicewp", "cliqwp", "rel", "rei", "pmxrgn", "o3vmr"]
base = {k: d["r8s_" + k].astype(np.float64) for k in keys}
nmx = d["r8s_nmxrgn"].astype(np.int32)
cosz = d["r8s_coszrs"].astype(np.float64)
co2 = d["r8s_co2mmr"].astype(np.float64)
solcon = d["solcon"].astype(np.float32)
alb = d["in_albedo"].astype(np.float32)
jul = d["in_julian"].astype(np.float32)
ncam = cosz.shape[0]
desc = [f"cam01_{i}" for i in range(ncam)]

day = np.flatnonzero(cosz > 0)
rng = np.random.default_rng(20261010)
syn = []


def add(src, cld, label, cz=None, lwp=40.0, iwp=15.0):
    c = {k: base[k][src].copy() for k in keys}
    cld = np.asarray(cld, np.float64)
    c["cld"] = cld
    cold = c["t"] < 263.16
    warm = c["t"] > 273.16
    c["cliqwp"] = np.where(cld > 0, np.where(cold, 0.0, lwp * (0.5 + rng.random(nz))), 0.0)
    c["cicewp"] = np.where(cld > 0, np.where(warm, 0.0, iwp * (0.5 + rng.random(nz))), 0.0)
    pmx, nm = cldovrlap(c["pint"][None], cld[None])
    c["pmxrgn"] = np.asarray(pmx)[0]
    syn.append((c, int(np.asarray(nm)[0]), cosz[src] if cz is None else cz, src, label))


def blocks(levels, values):
    cld = np.zeros(nz)
    for k, v in zip(levels, values):
        cld[k] = v
    return cld


for n in range(12):     # totwgt = 0 -> maximum overlap second pass
    src = day[n % day.size]
    lev = list(range(8, 8 + 2 * 10, 2))
    vals = rng.choice([0.3, 0.5, 0.7], size=10) if n % 2 else [0.5] * 10
    add(src, blocks(lev, vals), "fallback_pass2")
for n in range(16):     # > 15 qualifying configurations with tied weights
    src = day[(5 * n + 3) % day.size]
    lev = [10, 14, 18, 22, 26][: 4 + n % 2]
    pat = [[0.5, 0.5, 0.5, 0.4], [0.5, 0.5, 0.4, 0.6, 0.5], [0.3, 0.5, 0.5, 0.5], [0.5, 0.25, 0.5, 0.5, 0.75]][n % 4]
    add(src, blocks(lev, pat[: len(lev)]), "ties_findvalue")
for n in range(40):     # random multi-region clouds, quantized fractions (ties) or continuous, cld = 1 layers
    src = day[(7 * n + 1) % day.size]
    p = rng.uniform(0.15, 0.6)
    cl = rng.random(nz) < p
    if n % 3 == 0:
        v = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0], size=nz)
    elif n % 3 == 1:
        v = rng.uniform(0.01, 1.0, size=nz)
    else:
        v = np.where(rng.random(nz) < 0.3, 1.0, rng.choice([0.25, 0.5, 0.75], size=nz))
    cld = np.where(cl, v, 0.0)
    cld[0] = 0.0
    add(src, cld, "random")
for n in range(8):      # very low sun (exp-argument clamps) and an overcast column
    src = day[(11 * n + 2) % day.size]
    cld = np.where(rng.random(nz) < 0.4, rng.uniform(0.05, 1.0, nz), 0.0)
    add(src, cld, "low_sun", cz=[0.003, 0.02, 0.05, 0.1][n % 4])

ncol = ncam + len(syn)
with open(out, "wb") as f:
    f.write(np.asarray([ncol, nz, mode], "<i4").tobytes())
    f.write(np.asarray(d["in_p_top"], "<f4").reshape(1).tobytes())
    f.write(np.asarray(d["in_znu"], "<f4").tobytes())

    def col(c, nm, cz, co, sc, al, ju):
        f.write(np.concatenate([np.asarray(c[k], "<f8").ravel() for k in keys]).tobytes())
        f.write(np.asarray([nm], "<i4").tobytes())
        f.write(np.asarray([cz, co], "<f8").tobytes())
        f.write(np.asarray([sc, al, ju], "<f4").tobytes())

    for i in range(ncam):
        col({k: base[k][i] for k in keys}, nmx[i], cosz[i], co2[i], solcon[i], alb[i], jul[i])
    for c, nm, cz, src, label in syn:
        col(c, nm, cz, co2[src], solcon[src], alb[src], jul[src])
        desc.append(f"{label}_src{src}")
json.dump({"ncol": ncol, "ncam01": ncam, "mode": mode, "columns": desc}, open(cols_json, "w"), indent=0)
print("wrote", out, ncol, "columns", len(syn), "synthetic")
