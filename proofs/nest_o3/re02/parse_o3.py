"""RE02: nest ozone seam magnitude. arm1 = WRF nest o3rad (d01 parent profile level-wise), arm2 = port per-domain ozone."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "<USER_HOME>/wrf_gpu2_lanes/fid-q2/RE01")
from parse import SCAL  # noqa: E402

fx = np.load("<USER_HOME>/wrf_gpu2_lanes/fid-q2/RE01/re01_fixture.npz")
ncol, nz = fx["in_t"].shape
raw = np.frombuffer(Path("re02_output.bin").read_bytes(), np.float32)
assert tuple(raw[:2].view(np.int32)) == (ncol, nz)
pos = 2
def take(n):
    global pos
    a = raw[pos:pos + n]; pos += n; return a
o3w, o3p, arms = [], [], [{k: [] for k in ("hlw", "hlwc", "hsw", "hswc", "lwup", "lwupc", "lwdn", "lwdnc", "swup", "swupc",
                                             "swdn", "swdnc", *SCAL)} for _ in range(2)]
for _ in range(ncol):
    o3w.append(take(nz)); o3p.append(take(nz))
    for a in arms:
        for k in ("hlw", "hlwc", "hsw", "hswc"):
            a[k].append(take(nz))
        for k in ("lwup", "lwupc", "lwdn", "lwdnc", "swup", "swupc", "swdn", "swdnc"):
            a[k].append(take(nz + 2))
        for k, v in zip(SCAL, take(len(SCAL))):
            a[k].append(v)
assert pos == raw.size
o3w, o3p = np.asarray(o3w), np.asarray(o3p)
w, p = ({k: np.asarray(v) for k, v in a.items()} for a in arms)
same_as_re01 = all(np.array_equal(w[k], fx[f"A_{k}"]) for k in w)
rel = np.abs(o3p - o3w) / np.maximum(o3w, 1e-12)
res = {"arm1_bitwise_equals_RE01_A": bool(same_as_re01),
       "o3_rel_diff_max_by_level_band": {"k0-9": float(rel[:, :10].max()), "k10-29": float(rel[:, 10:30].max()),
                                          "k30-43": float(rel[:, 30:].max())},
       "o3_abs_diff_max_vmr": float(np.abs(o3p - o3w).max())}
for k, b in (("glw", (1, .05)), ("olr", (1, .05)), ("swdnb", (1, .05)), ("swupt", (1, .05)), ("lwdn", (1, .05)), ("swdn", (1, .05))):
    d = np.abs(p[k].astype(np.float64) - w[k])
    res[k] = {"max_abs": float(d.max()), "max_bound_ratio": float((d / (b[0] + b[1] * np.abs(w[k]))).max())}
for k in ("hlw", "hsw"):
    d = np.abs(p[k].astype(np.float64) - w[k])
    lev = np.unravel_index(np.argmax(d), d.shape)
    res[k] = {"max_abs_K_per_day": float(d.max() * 86400), "at_level": int(lev[1]),
              "max_bound_ratio": float((d / (1e-4 + 0.05 * np.abs(w[k]))).max())}
dom = fx["domain"]
for dname in ("d02", "d03"):
    m = dom == dname
    res[f"{dname}_hlw_max_K_per_day"] = float(np.abs(p["hlw"] - w["hlw"])[m].max() * 86400)
    res[f"{dname}_hsw_max_K_per_day"] = float(np.abs(p["hsw"] - w["hsw"])[m].max() * 86400)
Path("re02_summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
