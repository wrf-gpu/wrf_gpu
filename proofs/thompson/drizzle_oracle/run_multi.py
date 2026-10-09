"""LL01 multi-step MP-only: N repeated pristine mp_gt_driver steps vs N native C24 full-column steps on the same real
columns (no dynamics in either; integrates small per-step differences in drizzle production/fallout).
usage: run_multi.py <columns.npz> <dt> <nsteps> <out.npz>"""
import os, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import run_oracle as ro
EXE = Path("<USER_HOME>/wrf_gpu2_lanes/b-thompson/LL01/drv_multi/thompson_column_driver.exe")


def run_wrf(z, idx, dt, n, tag):
    ncol, nlev = len(idx), z["in_qv"].shape[1]
    fin, fout = ro.DRV / f"in_{tag}.bin", ro.DRV / f"out_{tag}.bin"
    with open(fin, "wb") as f:
        f.write(np.array([ncol, nlev], ">i4").tobytes()); f.write(np.array([dt], ">f4").tobytes())
        for v in ro.ORDER_IN:
            f.write(np.asarray(z[f"in_{v}"][idx].T, ">f4").tobytes(order="C"))
    subprocess.run(["taskset", "-c", "24", str(EXE), str(fin), str(fout), str(n)], cwd=ro.DRV, check=True, capture_output=True)
    raw = np.fromfile(fout, ">f4"); out, o = {}, 0
    for v in ro.ORDER_OUT:
        out[v] = raw[o:o + ncol * nlev].reshape(nlev, ncol).T.astype(np.float32); o += ncol * nlev
    for v in ("rainncv", "snowncv", "graupelncv", "sr", "rainnc"):
        out[v] = raw[o:o + ncol].astype(np.float32); o += ncol
    return out


def run_port(z, idx, dt, n):
    os.environ.update(ro.C24)
    import jax, jax.numpy as jnp
    assert jax.devices()[0].platform == "cpu"
    from gpuwrf.physics import thompson_column as tc
    from gpuwrf.kernels import phys_thompson_full as full
    jax.clear_caches()
    f = lambda v: jnp.asarray(z[f"in_{v}"][idx], jnp.float32)  # noqa: E731
    T, p, qv = f("th") * f("pii"), f("p"), f("qv")
    zero = jnp.zeros_like(qv)
    s = tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"), Ni=f("ni"), Nr=f("nr"),
                               Ns=zero, Ng=zero, T=T, p=p, rho=tc.density_from_pressure_temperature(p, T, qv),
                               dz=f("dz8w"), w=f("w"))
    step = jax.jit(lambda st: full.full_column(st, float(dt), interpret=True))
    rain = np.zeros(len(idx), np.float64)
    for i in range(n):
        # WRF mp_gt_driver re-derives rho from (p, t, qv) every call; the port caller does the same each step.
        s = s.replace(rho=tc.density_from_pressure_temperature(s.p, s.T, s.qv))
        s, ppt = step(s)
        rain += np.asarray(ppt["rain"], np.float64) + np.asarray(ppt["snow"], np.float64) + \
            np.asarray(ppt["graupel"], np.float64) + np.asarray(ppt["ice"], np.float64)
    g = lambda a: np.asarray(a, np.float32)  # noqa: E731
    pii = np.asarray(z["in_pii"][idx], np.float32)
    return {"qv": g(s.qv), "qc": g(s.qc), "qr": g(s.qr), "qi": g(s.qi), "qs": g(s.qs), "qg": g(s.qg), "ni": g(s.Ni),
            "nr": g(s.Nr), "th": g(s.T) / pii, "rainnc": rain.astype(np.float32), "rho": g(s.rho)}


def main():
    z = dict(np.load(sys.argv[1])); dt, n = float(sys.argv[2]), int(sys.argv[3])
    idx = np.flatnonzero(z["dt"] == dt); tag = f"multi_dt{int(dt)}_n{n}"
    t0 = time.time(); w = run_wrf(z, idx, dt, n, tag); print("wrf done", tag, len(idx), "%.0f s" % (time.time() - t0), flush=True)
    t0 = time.time(); p = run_port(z, idx, dt, n); print("port done", tag, "%.0f s" % (time.time() - t0), flush=True)
    np.savez_compressed(sys.argv[4], idx=idx, **{f"wrf_{k}": v for k, v in w.items()}, **{f"port_{k}": v for k, v in p.items()})


if __name__ == "__main__":
    main()
