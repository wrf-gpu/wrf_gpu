import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "proofs/v034/camuw_oracle")
import gpuwrf.physics.bl_camuw as cu
import make_columns as mc

def rec_from(rows):
    f = lambda a: np.asarray(a, np.float32)
    (hfx, qfx, ust, ht), u, v, th, p, t, exn, rho, qv, qc, qi, nc, ni, cf, lw, ws, z, p8w, zw = rows
    kx = len(u)
    return dict(dt=np.float32(60.0), u_phy=f(u), v_phy=f(v), th_phy=f(th), rho=f(rho), qv=f(qv), qc=f(qc), qi=f(qi),
                qnc=f(nc), qni=f(ni), p_phy=f(p), p8w=f(p8w), z=f(z), z_at_w=f(zw), t_phy=f(t), cldfra=f(cf),
                rthratenlw=f(lw), exner=f(exn), wsedl=f(ws), hfx=np.float32(hfx), qfx=np.float32(qfx),
                ustar=np.float32(ust), ht=np.float32(ht), kvm3d=np.zeros(kx+1, np.float32), kvh3d=np.zeros(kx+1, np.float32),
                tauresx2d=np.float32(0), tauresy2d=np.float32(0), first_step=np.bool_(True))

fn = jax.jit(lambda kw: cu.camuw_column(**kw))
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
target = sys.argv[1]
for trial in range(400):
    if target == "merge_up":
        a1 = rng.uniform(-0.003, -0.0002); z1 = rng.uniform(300, 1200); d = rng.uniform(50, 400); b = rng.uniform(0.00002, 0.003)
        a2 = rng.uniform(-0.004, -0.0001); z2 = z1 + d + rng.uniform(200, 1500); hfx = rng.uniform(50, 450)
        def th(z, a1=a1, z1=z1, d=d, b=b, a2=a2, z2=z2):
            if z < z1: return 300 + a1 * z
            t1 = 300 + a1 * z1
            if z < z1 + d: return t1 + b * (z - z1)
            t2 = t1 + b * d
            if z < z2: return t2 + a2 * (z - z1 - d)
            return t2 + a2 * (z2 - z1 - d) + 0.005 * (z - z2)
        name, rows = mc.column("s", psfc=100600.0, ht=0.0, theta_fn=th, q_fn=lambda z: max(0.010*np.exp(-z/2500.0), 2e-5),
                               u_fn=lambda z: 4.0 + 0.001*z, v_fn=lambda z: 1.0, hfx=hfx, qfx=8e-5, ust=0.35)
        params = dict(a1=a1, z1=z1, d=d, b=b, a2=a2, z2=z2, hfx=hfx)
    else:
        top = rng.uniform(2.0, 18.0); qcv = rng.uniform(5e-5, 8e-4); lap = rng.uniform(0.002, 0.03); hfx = -rng.uniform(0.5, 30)
        qv0 = rng.uniform(0.003, 0.009); qdrop = rng.uniform(0.0, 0.002); lwv = -rng.uniform(5e-5, 2e-3); ust = rng.uniform(0.03, 0.3)
        name, rows = mc.column("s", psfc=101700.0, ht=10.0, theta_fn=mc.ml_theta(279.0, 300.0, lap, 0.005),
                               q_fn=lambda z, qv0=qv0, qdrop=qdrop: qv0 if z < 6.0 else max(qv0 - qdrop, 2e-5),
                               qc_fn=lambda z, top=top, qcv=qcv: qcv if z < top else 0.0,
                               lw_fn=lambda z, top=top, lwv=lwv: lwv if z < top else -1.0e-5,
                               u_fn=lambda z: 1.0 + 0.002*z, v_fn=lambda z: 0.3, hfx=hfx, qfx=-5e-7, ust=ust)
        params = dict(top=top, qcv=qcv, lap=lap, hfx=hfx, qv0=qv0, qdrop=qdrop, lwv=lwv, ust=ust)
    cu._CENSUS = {}
    fn = jax.jit(lambda kw: cu.camuw_column(**kw))
    jax.block_until_ready(fn(rec_from(rows))); jax.effects_barrier()
    c = cu._CENSUS
    key = "zisocl_merge_up" if target == "merge_up" else "srcl_surface"
    if c.get(key, 0) > 0:
        print("FOUND", trial, params, c); break
else:
    print("none")
