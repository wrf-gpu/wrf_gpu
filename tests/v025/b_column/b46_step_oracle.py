"""B46 one-step CPU oracle: GWDO folded into RUBLTEN/RVBLTEN from the pre-PBL state.

Real PROD d01 (s0_case_20260725, initial carry), CPU, eager. Two calls of
_physics_step_forcing (source-leaf PBL path, capture_first_interval=True):
gwd_opt=1 vs gwd_opt=0, identical otherwise. The opmode gwdo_tendencies binding is
wrapped to record its input state. Checks:
  (1) RUBLTEN/RVBLTEN(gwd=1) - (gwd=0) == pristine bl_gwdo_run (REAL, unmodified, WRF
      gfortran flags) on WRF phy_prep columns of the recorded state (A-grid, m/s^2);
  (2) the recorded input state is the pre-PBL state: u/v/theta/qv equal the state the
      PBL slot received (= the gwd=0 run's pbl_entry), not the post-MYNN state;
  (3) next_state u/v are bitwise equal with and without GWDO (no step-entry Euler add).
usage: b46_step_oracle.py <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CASE = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")


def phy_prep_columns(state, fnm, fnp):
    """WRF phy_prep columns (float32 NumPy, same formulas as build_gwdo_oracle) from State leaves."""
    F = np.float32
    g = lambda a: np.asarray(a, np.float32)
    theta, qv, p, ph, u, v = (g(state.theta), g(state.qv), g(state.p), g(state.ph), g(state.u), g(state.v))
    r_d, r_v = F(287.), F(461.6)
    rcp = r_d / (F(7.) * r_d / F(2.))
    th = theta / (F(1.) + r_v / r_d * qv)
    pi_phy = np.power(p / F(100000.), rcp)
    z_at_w = ph / F(9.81)
    z = F(.5) * (z_at_w[:-1] + z_at_w[1:])
    nz = p.shape[0]
    fnm, fnp = g(fnm), g(fnp)
    p8w = np.empty((nz + 1,) + p.shape[1:], F)
    for k in range(1, nz):
        p8w[k] = fnm[k] * p[k] + fnp[k] * p[k - 1]
    w1 = (z_at_w[0] - z[1]) / (z[0] - z[1])
    p8w[0] = w1 * p[0] + (F(1.) - w1) * p[1]
    w1 = (z_at_w[nz] - z[nz - 2]) / (z[nz - 1] - z[nz - 2])
    p8w[nz] = np.exp(w1 * np.log(p[nz - 1]) + (F(1.) - w1) * np.log(p[nz - 2]))
    flat = lambda a: np.ascontiguousarray(a.reshape(a.shape[0], -1), F)
    return dict(uproj=flat(F(.5) * (u[:, :, :-1] + u[:, :, 1:])), vproj=flat(F(.5) * (v[:, :-1, :] + v[:, 1:, :])),
                t1=flat(th * pi_phy), q1=flat(qv), prsl=flat(p), prslk=flat(pi_phy), zl=flat(z), prsi=flat(p8w))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import gpuwrf.contracts.state as state_contract
    state_contract._gpu_device = lambda: jax.devices("cpu")[0]  # CPU replay (repo pattern, cf. v0234 CPU A/Bs)
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
    from gpuwrf.runtime import operational_mode as om
    from build_gwdo_oracle import build_driver, run_case
    t0 = time.perf_counter()
    loaded = _load_domains(NestedPipelineConfig(CASE, out / "unused", out / "proof", hours=1, max_dom=2),
                           ("d01", "d02"))
    hierarchy, bundles, meta, start, dts, carries = loaded
    nl = bundles["d01"].namelist
    carry = carries["d01"]
    rec = dict(load_s=time.perf_counter() - t0, rad_rk_tendf=int(nl.rad_rk_tendf), gwd_opt=int(nl.gwd_opt),
               bl=int(nl.bl_pbl_physics), dt=float(nl.dt_s), statics=nl.gwdo_statics is not None)
    print(rec, flush=True)
    assert rec["rad_rk_tendf"] != 0 and rec["gwd_opt"] == 1 and rec["statics"]

    seen = {}
    real, real_mynn = om.gwdo_tendencies, om.mynn_adapter_with_source_leaves

    def spy(state, dt, statics, grid, **kwargs):
        res = real(state, dt, statics, grid, **kwargs)
        seen["state"], seen["res"] = state, res
        return res

    def spy_mynn(state, *a, **k):
        seen.setdefault("mynn_in", state)
        return real_mynn(state, *a, **k)

    om.gwdo_tendencies, om.mynn_adapter_with_source_leaves = spy, spy_mynn
    runs = {}
    for gwd in (1, 0):
        t1 = time.perf_counter()
        with jax.disable_jit():
            forcing, record = om._physics_step_forcing(
                carry, replace(nl, gwd_opt=gwd), jnp.asarray(float(nl.dt_s)), run_radiation=False,
                first_timestep=True, capture_first_interval=True)
        runs[gwd] = (forcing, record)
        print("gwd", gwd, "step s", round(time.perf_counter() - t1, 1), flush=True)
    om.gwdo_tendencies, om.mynn_adapter_with_source_leaves = real, real_mynn
    assert "state" in seen, "GWDO slot did not call gwdo_tendencies (fold path not taken)"

    f1, r1 = runs[1]
    f0, r0 = runs[0]
    st, mi = seen["state"], seen["mynn_in"]
    np.savez(out / "raw.npz", rub1=np.asarray(r1.rublten), rvb1=np.asarray(r1.rvblten),
             rub0=np.asarray(r0.rublten), rvb0=np.asarray(r0.rvblten),
             u1=np.asarray(f1.state.u), u0=np.asarray(f0.state.u), v1=np.asarray(f1.state.v), v0=np.asarray(f0.state.v),
             ru1=np.asarray(f1.dry_tendencies.ru_tendf), ru0=np.asarray(f0.dry_tendencies.ru_tendf),
             **{f"st_{n}": np.asarray(getattr(st, n)) for n in ("u", "v", "theta", "qv", "p", "ph")},
             **{f"mi_{n}": np.asarray(getattr(mi, n)) for n in ("u", "v", "theta", "qv", "p", "ph")})
    d_ru = np.asarray(r1.rublten) - np.asarray(r0.rublten)
    d_rv = np.asarray(r1.rvblten) - np.asarray(r0.rvblten)
    nz, ny, nx = np.asarray(st.theta).shape
    # (2) input = pre-PBL state: bitwise the state the MYNN adapter received.
    gwdo_input_is_pbl_entry = all(np.array_equal(np.asarray(getattr(st, n)), np.asarray(getattr(mi, n)))
                                  for n in ("u", "v", "theta", "qv", "p", "ph"))
    # (3) no Euler add
    uv_equal = bool(np.array_equal(np.asarray(f1.state.u), np.asarray(f0.state.u))
                    and np.array_equal(np.asarray(f1.state.v), np.asarray(f0.state.v)))
    # (1) pristine on WRF phy_prep columns of the recorded state
    gwdo = nl.gwdo_statics
    cols = phy_prep_columns(st, nl.grid.metrics.fnm, nl.grid.metrics.fnp)
    stat = dict(var=gwdo.var, con=gwdo.oc1, oa1=gwdo.oa1, oa2=gwdo.oa2, oa3=gwdo.oa3, oa4=gwdo.oa4,
                ol1=gwdo.ol1, ol2=gwdo.ol2, ol3=gwdo.ol3, ol4=gwdo.ol4, sina=gwdo.sina, cosa=gwdo.cosa,
                dxmeter=gwdo.dxmeter)
    stat = {k: np.ascontiguousarray(np.asarray(v, np.float32)) for k, v in stat.items()}
    env, _, _ = build_driver(HERE.parents[2], Path("<USER_HOME>/src/wrf_pristine/WRF"), out,
                             Path("<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"))
    wrf = run_case(out, env, cols, stat, float(nl.dt_s), out / "pristine")
    e_ru = wrf["rublten"].reshape(nz, ny, nx).astype(np.float64)
    e_rv = wrf["rvblten"].reshape(nz, ny, nx).astype(np.float64)
    rec.update(
        gwdo_scale_u=float(np.abs(e_ru).max()), gwdo_scale_v=float(np.abs(e_rv).max()),
        fold_err_u=float(np.abs(d_ru - e_ru).max()), fold_err_v=float(np.abs(d_rv - e_rv).max()),
        active_columns=int(((np.abs(e_ru) + np.abs(e_rv)).max(0) > 0).sum()),
        uv_bitwise_equal_with_without_gwdo=uv_equal, gwdo_input_is_pbl_entry_state=gwdo_input_is_pbl_entry,
        ru_tendf_changed=bool(np.abs(np.asarray(f1.dry_tendencies.ru_tendf) - np.asarray(f0.dry_tendencies.ru_tendf)).max() > 0))
    (out / "b46_step_oracle.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    # Gate: REAL-vs-f64 rounding only (5e-5 of the GWDO scale), WRF input state, no Euler add.
    assert rec["fold_err_u"] <= 5e-5 * rec["gwdo_scale_u"] and rec["fold_err_v"] <= 5e-5 * rec["gwdo_scale_v"], rec
    assert rec["gwdo_input_is_pbl_entry_state"], "GWDO must read the pre-PBL (phy_prep) state"
    assert rec["uv_bitwise_equal_with_without_gwdo"], "GWDO must not add a step-entry Euler increment"
    assert rec["ru_tendf_changed"], "GWDO drag must reach ru/rv_tendf"


if __name__ == "__main__":
    main()
