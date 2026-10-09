"""LL01: one pristine mp_gt_driver step (TH08 driver, real 44-level 0227 columns, per-domain dt) vs the product
native REAL Thompson full column (release-default C24 flags, interpret Pallas on CPU) on the SAME inputs.
usage: run_oracle.py <columns.npz> <out.npz>"""
import os, subprocess, sys
from pathlib import Path
import numpy as np
ORDER_IN = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "w", "dz8w")
ORDER_OUT = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th")
EXE = Path("<USER_HOME>/wrf_gpu2_lanes/b-thompson/LL01/drv_src/thompson_column_driver.exe")
DRV = Path("<USER_HOME>/wrf_gpu2_lanes/b-thompson/LL01/drv")   # own cwd: tables computed once, written here, reused
C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0",
           GPUWRF_THOMPSON_IMPLICIT_SED="0", GPUWRF_THOMPSON_FP32="0")


def run_wrf(z, idx, dt, tag):
    ncol, nlev = len(idx), z["in_qv"].shape[1]
    fin, fout = DRV / f"in_{tag}.bin", DRV / f"out_{tag}.bin"
    with open(fin, "wb") as f:
        f.write(np.array([ncol, nlev], ">i4").tobytes()); f.write(np.array([dt], ">f4").tobytes())
        for v in ORDER_IN:
            f.write(np.asarray(z[f"in_{v}"][idx].T, ">f4").tobytes(order="C"))
    subprocess.run(["taskset", "-c", "24", str(EXE), str(fin), str(fout)], cwd=DRV, check=True, capture_output=True)
    raw = np.fromfile(fout, ">f4"); out, o = {}, 0
    for v in ORDER_OUT:
        out[v] = raw[o:o + ncol * nlev].reshape(nlev, ncol).T.astype(np.float32); o += ncol * nlev
    for v in ("rainncv", "snowncv", "graupelncv", "sr"):
        out[v] = raw[o:o + ncol].astype(np.float32); o += ncol
    return out


def run_port(z, idx, dt):
    os.environ.update(C24)
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
    out, ppt = full.full_column(s, float(dt), interpret=True)
    g = lambda a: np.asarray(a, np.float32)  # noqa: E731
    pii = np.asarray(z["in_pii"][idx], np.float32)
    return {"qv": g(out.qv), "qc": g(out.qc), "qr": g(out.qr), "qi": g(out.qi), "qs": g(out.qs), "qg": g(out.qg),
            "ni": g(out.Ni), "nr": g(out.Nr), "th": g(out.T) / pii, "rainncv": g(ppt["rain"]), "snowncv": g(ppt["snow"]),
            "graupelncv": g(ppt["graupel"]), "rho": g(s.rho)}


def main():
    z = dict(np.load(sys.argv[1])); DRV.mkdir(exist_ok=True)
    for name in ("CCN_ACTIVATE.BIN",):
        if not (DRV / name).exists():
            (DRV / name).symlink_to("<USER_HOME>/src/wrf_pristine/WRF/test/em_real/oracle_run/" + name)
    res = {}
    for dt in sorted(set(z["dt"].tolist())):
        idx = np.flatnonzero(z["dt"] == dt); tag = f"dt{int(dt)}"
        w = run_wrf(z, idx, dt, tag); print("wrf done", tag, len(idx), flush=True)
        p = run_port(z, idx, dt); print("port done", tag, flush=True)
        for k, v in w.items():
            res.setdefault(f"wrf_{k}", np.zeros((len(z["dt"]),) + v.shape[1:], np.float32))[idx] = v
        for k, v in p.items():
            res.setdefault(f"port_{k}", np.zeros((len(z["dt"]),) + v.shape[1:], np.float32))[idx] = v
    np.savez_compressed(sys.argv[2], **res)


if __name__ == "__main__":
    main()
