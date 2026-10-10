"""SWIDX: parse swidx_output.bin (+ the input) into one npz fixture.  Usage: parse_output.py in.bin out.bin cols.json fx.npz"""
import json
import sys

import numpy as np

fin, fout, cols_json, fx = sys.argv[1:5]
keys = ["q1", "cld", "pmid", "pint", "t", "cicewp", "cliqwp", "rel", "rei", "pmxrgn", "o3vmr"]


class R:
    def __init__(self, p):
        self.b = open(p, "rb").read(); self.p = 0

    def take(self, dt, n):
        dt = np.dtype(dt); a = np.frombuffer(self.b, dt, n, self.p).copy(); self.p += dt.itemsize * n; return a


ri = R(fin)
ncol, nz, mode = ri.take("<i4", 3)
p_top = ri.take("<f4", 1)[0]; znu = ri.take("<f4", nz)
nzp = nz + 1
sizes = [nz, nz, nz, nzp, nz, nz, nz, nz, nz, nzp, nz]
acc = {f"in_{k}": [] for k in keys}
for n in ("in_nmxrgn", "in_coszrs", "in_co2mmr", "in_solcon", "in_albedo", "in_julian"):
    acc[n] = []
for _ in range(ncol):
    for k, s in zip(keys, sizes):
        acc[f"in_{k}"].append(ri.take("<f8", s))
    acc["in_nmxrgn"].append(ri.take("<i4", 1)[0])
    cz, co = ri.take("<f8", 2); acc["in_coszrs"].append(cz); acc["in_co2mmr"].append(co)
    sc, al, ju = ri.take("<f4", 3); acc["in_solcon"].append(sc); acc["in_albedo"].append(al); acc["in_julian"].append(ju)
assert ri.p == len(ri.b)
ro = R(fout)
ncol2, nz2, mode2, mxaerl = ro.take("<i4", 4)
assert (ncol2, nz2, mode2) == (ncol, nz, mode)
m_hybi = ro.take("<f4", 29)
aerc = ro.take("<f4", 29 * 13).reshape(13, 29).T.copy()
names_p = ["qrs", "qrscs", "fsup", "fsupc", "fsdn", "fsdnc", "fsdndir", "fsdndif", "fsdncdir", "fsdncdif", "tauxcl",
           "tauxci"]
sz_p = [nz, nz] + [nzp] * 8 + [nz, nz]
names_s = ["fsns", "fsntoa", "fsntoac", "fsds", "fsdsdir", "fsdsdif", "sols", "soll", "solsd", "solld", "solin"]
for n in ["rh", "aerosol", "nmx_after", "pmx1_after"] + names_p + names_s:
    acc["o_" + n] = []
for _ in range(ncol):
    acc["o_rh"].append(ro.take("<f8", nz))
    acc["o_aerosol"].append(ro.take("<f8", nz * 13).reshape(13, nz).T.copy())
    for n, s in zip(names_p, sz_p):
        acc["o_" + n].append(ro.take("<f8", s))
    for n, v in zip(names_s, ro.take("<f8", len(names_s))):
        acc["o_" + n].append(v)
    acc["o_nmx_after"].append(ro.take("<i4", 1)[0])
    acc["o_pmx1_after"].append(ro.take("<f8", 1)[0])
assert ro.p == len(ro.b)
outd = {k: np.asarray(v) for k, v in acc.items()}
outd.update(mode=np.int32(mode), mxaerl=np.int32(mxaerl), m_hybi=m_hybi, aerc=aerc, p_top=np.float32(p_top), znu=znu)
meta = json.load(open(cols_json))
outd["columns"] = np.asarray(meta["columns"])
outd["ncam01"] = np.int32(meta["ncam01"])
np.savez_compressed(fx, **outd)
print("fixture", fx, ncol, "columns, mode", mode, "mxaerl", mxaerl)
