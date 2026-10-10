"""Real-data multi-step smoke: CAM-UW on ALL 42x42 Swiss CPU-WRF columns (12z), 10 steps of dt=60 s,
state evolved by the PBL tendencies, KVM/KVH/TAURES carried as REAL. Checks finiteness + sanity."""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np, netCDF4
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "proofs/v034/camuw_oracle")
import make_real_columns as mr
from gpuwrf.physics.bl_camuw import camuw_column
d = netCDF4.Dataset(mr.SRC.format(hh="12")); d.set_auto_mask(False)
cols = []
for j in range(42):
    for i in range(42):
        rows, kx = mr.column(d, j, i); cols.append(rows)
F = np.float32
def stack(idx): return np.stack([np.asarray(c[idx], F) for c in cols])
sc = np.stack([np.asarray(c[0], F) for c in cols])
names = ["u_phy","v_phy","th_phy","p_phy","t_phy","exner","rho","qv","qc","qi","qnc","qni","cldfra","rthratenlw","wsedl","z","p8w","z_at_w"]
inp = {n: stack(k + 1) for k, n in enumerate(names)}
inp.update(hfx=sc[:, 0], qfx=sc[:, 1], ustar=sc[:, 2], ht=sc[:, 3])
n = len(cols)
inp.update(kvm3d=np.zeros((n, kx + 1), F), kvh3d=np.zeros((n, kx + 1), F), tauresx2d=np.zeros(n, F), tauresy2d=np.zeros(n, F))
f = jax.jit(jax.vmap(lambda kw: camuw_column(dt=jnp.float32(60.0), first_step=False, **kw)))
rep = []
for step in range(10):
    t0 = time.time(); o = jax.tree_util.tree_map(np.asarray, f(inp)); dt_s = time.time() - t0
    fin = all(bool(np.all(np.isfinite(v))) for v in o.values())
    rep.append(dict(step=step + 1, wall_s=round(dt_s, 2), finite=fin, max_kvh=float(o["kvh3d"].max()),
                    max_abs_rthblten=float(np.abs(o["rthblten"]).max()), max_abs_rublten=float(np.abs(o["rublten"]).max()),
                    pblh_range=[float(o["pblh"].min()), float(o["pblh"].max())], min_qv_after=float((inp["qv"] + 60 * o["rqvblten"]).min())))
    print(rep[-1], flush=True)
    for a, b in (("u_phy", "rublten"), ("v_phy", "rvblten"), ("th_phy", "rthblten"), ("qv", "rqvblten"), ("qc", "rqcblten"), ("qi", "rqiblten"), ("qni", "rqniblten")):
        inp[a] = (inp[a] + F(60.0) * o[b]).astype(F)
    for a in ("qv", "qc", "qi", "qni"): inp[a] = np.maximum(inp[a], 0).astype(F)
    inp["t_phy"] = (inp["th_phy"] * inp["exner"]).astype(F)
    inp.update(kvm3d=o["kvm3d"], kvh3d=o["kvh3d"], tauresx2d=o["tauresx2d"], tauresy2d=o["tauresy2d"])
json.dump(dict(ncol=n, kx=kx, steps=rep), open("<USER_HOME>/wrf_gpu2_lanes/o1-camuw/swiss_smoke.json", "w"), indent=1)
