"""Multi-step daytime column integration: MYNN with EDMF mass-flux ON vs OFF.

Characterizes the CUMULATIVE near-surface qv effect of the mass-flux nonlocal
transport that the single-step oracle proves is missing. Holds surface fluxes
fixed at the 12z daytime-land values (a frozen-forcing column experiment) and
integrates the MYNN column kernel for a daytime window, comparing the lowest-level
qv trajectory between edmf=True and edmf=False.

This is an fp64 JAX experiment (WRF's fp32 quantizes the per-step MF Dqv at ~3e-9,
too coarse to resolve the cumulative trend). The single-step s_awqv arrays are
separately proven WRF-faithful in mf_oracle_compare.json (<0.5% rel err).

CPU-only. Run:
  JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 taskset -c 0-3 python3 integration_oracle.py
"""
import json
import os
import sys

import numpy as np

ROOT = "/home/enric/src/wrf_gpu2/.claude/worktrees/agent-afd276c1c17aa32e5"
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, step_mynn_pbl_column  # noqa: E402
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes  # noqa: E402

COL = os.path.join(ROOT, "proofs/mynn_edmf/column_d03_12z.json")
OUT = os.path.join(ROOT, "proofs/mynn_edmf/integration_mf_vs_ed.json")


def build(c):
    pr = c["profiles"]
    A = lambda k: jnp.array(pr[k], dtype=jnp.float64)[None, :]
    z = jnp.zeros_like(A("th"))
    st = MynnPBLColumnState(
        A("u"), A("v"), A("w"), A("th"), A("qv"), 0.5 * A("qke"),
        A("p"), A("rho"), A("dz"), z, z, z)
    su = c["surface"]
    s1 = lambda x: jnp.array([x], dtype=jnp.float64)
    sfc = SurfaceFluxes(
        ustar=s1(su["ust"]), theta_flux=s1(su["flt"]), qv_flux=s1(su["flqv"]),
        tau_u=s1(0.0), tau_v=s1(0.0),
        rhosfc=s1(su["psfc"] / (287.0 * su["tsk"])), fltv=s1(su["fltv"]))
    return st, sfc


def integrate(st, sfc, dt, nsteps, edmf):
    step = jax.jit(lambda s: step_mynn_pbl_column(s, dt, surface=sfc, edmf=edmf, dx=1000.0))
    qv0_hist = [float(st.qv[0, 0])]
    qv1_hist = [float(st.qv[0, 1])]
    for _ in range(nsteps):
        st = step(st)
        qv0_hist.append(float(st.qv[0, 0]))
        qv1_hist.append(float(st.qv[0, 1]))
    return st, np.array(qv0_hist), np.array(qv1_hist)


def main():
    c = json.load(open(COL))
    dt = c["config"]["delt"]
    minutes = 120.0  # 2-hour frozen-forcing daytime window
    nsteps = int(minutes * 60 / dt)

    st0, sfc = build(c)
    _, qv0_ed, qv1_ed = integrate(st0, sfc, dt, nsteps, edmf=False)
    _, qv0_mf, qv1_mf = integrate(st0, sfc, dt, nsteps, edmf=True)

    qv0_init = qv0_ed[0]
    res = {
        "window_min": minutes,
        "nsteps": nsteps,
        "dt": dt,
        "qv0_init_kgkg": qv0_init,
        "lev0": {
            "qv_final_ED": float(qv0_ed[-1]),
            "qv_final_MF": float(qv0_mf[-1]),
            "MF_minus_ED_kgkg": float(qv0_mf[-1] - qv0_ed[-1]),
        },
        "lev1": {
            "qv_final_ED": float(qv1_ed[-1]),
            "qv_final_MF": float(qv1_mf[-1]),
            "MF_minus_ED_kgkg": float(qv1_mf[-1] - qv1_ed[-1]),
        },
        "equiv_t2_qair_dry_bias_kgkg": -0.001739,
        "note": (
            "MF_minus_ED > 0 at the surface means the missing mass flux was "
            "leaving our near-surface TOO DRY (consistent with the equiv-T2 "
            "qair dry bias of -1.74e-3 kg/kg). Magnitudes are a frozen-forcing "
            "single-column proxy, not the full coupled GPU run."
        ),
    }
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"{nsteps} steps over {minutes} min, dt={dt}s")
    print(f"lev0 qv  init={qv0_init:.6e}")
    print(f"     ED  final={qv0_ed[-1]:.6e}  MF final={qv0_mf[-1]:.6e}  "
          f"MF-ED={qv0_mf[-1] - qv0_ed[-1]:+.4e}")
    print(f"lev1 qv  ED  final={qv1_ed[-1]:.6e}  MF final={qv1_mf[-1]:.6e}  "
          f"MF-ED={qv1_mf[-1] - qv1_ed[-1]:+.4e}")
    print(f"(equiv-T2 lev0 qair dry bias = -1.74e-3 kg/kg)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
