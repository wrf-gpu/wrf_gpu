"""JAX MYNN-EDMF column oracle: drives dmp_mf_columns on the real d03 12z land
column and compares s_aw / s_awqv / s_awqt / s_awqc against the WRF Fortran oracle.

CPU-only, fp64. Run:
  JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 taskset -c 0-3 \
    /path/to/python proofs/mynn_edmf/jax_oracle.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics.mynn_edmf import dmp_mf_columns, XLVCP  # noqa: E402

COL = os.path.join(ROOT, "proofs/mynn_edmf/column_d03_12z.json")
FORT = os.path.join(ROOT, "proofs/mynn_edmf/fortran_oracle/oracle_out.txt")
OUT = os.path.join(ROOT, "proofs/mynn_edmf/mf_oracle_compare.json")


def parse_fort(fn):
    out = {}
    for line in open(fn):
        line = line.strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if "," in v:
            out[k] = np.array([float(x) for x in v.split(",")])
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main():
    c = json.load(open(COL))
    pr = c["profiles"]
    su = c["surface"]
    nz = c["meta"]["nz"]

    arr = lambda k: jnp.array(pr[k], dtype=jnp.float64)[None, :]  # (1, nz)
    th = arr("th")
    qv = arr("qv")
    qc = arr("qc")
    qi = arr("qi")
    p = arr("p")
    exner = arr("exner")
    rho = arr("rho")
    dz = arr("dz")
    u = arr("u")
    v = arr("v")
    w = arr("w")
    qke = arr("qke")

    # WRF specific contents: sqv=qv/(1+qv), etc. (mynnedmf_pre_run)
    sqv = qv / (1.0 + qv)
    sqc = qc / (1.0 + qv)
    sqi = qi / (1.0 + qv)
    sqw = sqv + sqc + sqi
    thl = th - XLVCP / exner * sqc
    thv = th * (1.0 + 0.608 * sqv)

    zw = jnp.concatenate(
        [jnp.zeros((1, 1)), jnp.cumsum(dz, axis=-1)], axis=-1
    )  # (1, nz+1)

    s1 = lambda x: jnp.array([x], dtype=jnp.float64)
    res = dmp_mf_columns(
        sqw, sqv, sqc, u, v, w, th, thl, thv, arr("tk"), qke,
        p, exner, rho, dz, zw,
        ust=s1(su["ust"]), flt=s1(su["flt"]), fltv=s1(su["fltv"]),
        flq=s1(su["flq"]), flqv=s1(su["flqv"]),
        pblh=s1(su["pblh"]), ts=s1(su["tsk"]),
        dx=su["dx"], xland=s1(su["xland"]), dt=c["config"]["delt"],
    )

    jax_saw = np.asarray(res["s_aw"][0])
    jax_sawqv = np.asarray(res["s_awqv"][0])
    jax_sawqt = np.asarray(res["s_awqt"][0])
    jax_sawqc = np.asarray(res["s_awqc"][0])
    print("JAX active:", float(res["active"][0]), " maxmf:", float(res["maxmf"][0]))
    print("JAX s_aw   k0..6:", jax_saw[:7])
    print("JAX s_awqv k0..6:", jax_sawqv[:7])

    fo = parse_fort(FORT)
    f_saw = fo["edmf_s_aw"]
    f_sawqv = fo["edmf_s_awqv"]
    f_sawqt = fo["edmf_s_awqt"]
    f_sawqc = fo["edmf_s_awqc"]
    print("WRF s_aw   k0..6:", f_saw[:7])
    print("WRF s_awqv k0..6:", f_sawqv[:7])

    def relerr(a, b):
        scale = max(np.max(np.abs(b)), 1e-30)
        return float(np.max(np.abs(a - b)) / scale)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    cmp = {
        "s_aw": {"max_abs_wrf": float(np.max(np.abs(f_saw))),
                 "rel_max_err": relerr(jax_saw[:len(f_saw)], f_saw),
                 "rmse": rmse(jax_saw[:len(f_saw)], f_saw)},
        "s_awqv": {"max_abs_wrf": float(np.max(np.abs(f_sawqv))),
                   "rel_max_err": relerr(jax_sawqv[:len(f_sawqv)], f_sawqv),
                   "rmse": rmse(jax_sawqv[:len(f_sawqv)], f_sawqv)},
        "s_awqt": {"rel_max_err": relerr(jax_sawqt[:len(f_sawqt)], f_sawqt)},
        "s_awqc": {"rel_max_err": relerr(jax_sawqc[:len(f_sawqc)], f_sawqc)},
    }
    # predeclared tolerance: relative max error <= 5% (single-precision WRF vs
    # fp64 JAX, plume scheme has tanh/exp/condensation nonlinearity).
    TOL = 0.05
    cmp["TOL_rel"] = TOL
    cmp["PASS_s_aw"] = bool(cmp["s_aw"]["rel_max_err"] <= TOL)
    cmp["PASS_s_awqv"] = bool(cmp["s_awqv"]["rel_max_err"] <= TOL)
    cmp["jax_active"] = float(res["active"][0])
    cmp["jax_s_aw"] = jax_saw.tolist()
    cmp["jax_s_awqv"] = jax_sawqv.tolist()
    cmp["wrf_s_aw"] = f_saw.tolist()
    cmp["wrf_s_awqv"] = f_sawqv.tolist()

    with open(OUT, "w") as fh:
        json.dump(cmp, fh, indent=2)
    print()
    print(f"s_aw   rel_max_err = {cmp['s_aw']['rel_max_err']:.4f}  PASS={cmp['PASS_s_aw']}")
    print(f"s_awqv rel_max_err = {cmp['s_awqv']['rel_max_err']:.4f}  PASS={cmp['PASS_s_awqv']}")
    print(f"s_awqt rel_max_err = {cmp['s_awqt']['rel_max_err']:.4f}")
    print(f"s_awqc rel_max_err = {cmp['s_awqc']['rel_max_err']:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
