"""CPU unit probe: MYNN qke cold-start divergence on the d02 replay (case
20260521_18z_l2) and the WRF-faithful cure.

Localizes the qke=0 step-1 singularity and verifies that seeding the WRF
``mym_initialize`` cold-start TKE profile (module_bl_mynnedmf.F:691 /
module_bl_mynnedmf.F:1331) on the replay adapter, when the parent wrfout carries
no QKE, keeps qke + the surface fluxes finite where qke=0 diverges.

CPU-only (no dycore, no GPU): runs surface_adapter -> mynn_adapter on the loaded
t0 state for a handful of steps and traces field maxima. The decisive arbiter is
the full GPU forecast (separate proof object).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gpuwrf.integration.d02_replay import build_l2_d02_replay_case  # noqa: E402
from gpuwrf.coupling.physics_couplers import surface_adapter, mynn_adapter  # noqa: E402

RUN_DIR = "<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260521_18z_l2_72h_20260522T133443Z"
DT = 12.0
STEPS = 8


def _max(a):
    a = np.asarray(a)
    return float(np.nanmax(np.abs(a))) if a.size else 0.0


def _fin(a):
    return bool(np.all(np.isfinite(np.asarray(a))))


def _trace(case, seed_qke):
    state = case.state
    if seed_qke is not None:
        state = state.replace(qke=seed_qke)
    grid = case.grid
    rows = []
    qke0 = np.asarray(state.qke)
    rows.append({
        "step": 0,
        "qke_max": _max(qke0),
        "qke_min": float(np.nanmin(qke0)),
        "qke_fin": _fin(qke0),
    })
    for s in range(1, STEPS + 1):
        state = surface_adapter(state, DT)
        ustar = np.asarray(state.ustar)
        thf = np.asarray(state.theta_flux)
        tau = np.asarray(state.tau_u)
        state = mynn_adapter(state, DT, grid)
        qke = np.asarray(state.qke)
        row = {
            "step": s,
            "qke_max": _max(qke),
            "qke_fin": _fin(qke),
            "ustar_max": _max(ustar),
            "ustar_fin": _fin(ustar),
            "theta_flux_max": _max(thf),
            "theta_flux_fin": _fin(thf),
            "tau_u_max": _max(tau),
            "tau_u_fin": _fin(tau),
            "theta_max": _max(state.theta),
            "theta_fin": _fin(state.theta),
            "u_fin": _fin(state.u),
            "v_fin": _fin(state.v),
            "all_fin": _fin(qke) and _fin(ustar) and _fin(thf) and _fin(tau)
            and _fin(state.theta) and _fin(state.u) and _fin(state.v),
        }
        rows.append(row)
        if not row["all_fin"]:
            break
    return rows


def _wrf_coldstart_qke(case):
    """WRF mym_initialize cold-start TKE seed (module_bl_mynnedmf.F:691).

    qke(kts) = 1.5 * max(ust,0.02)^2 * (b1*pmz)^(2/3); taper with height to a
    0.01 floor. Uses the surface ustar produced by one surface_adapter pass on
    the t0 state (matching WRF's surface-then-PBL call order).
    """
    B1 = 24.0
    PMZ = 1.0
    state = surface_adapter(case.state, DT)
    ust = np.asarray(state.ustar)  # (ny,nx)
    ph = np.asarray(state.ph_total)  # (nz+1,ny,nx)
    g = 9.81
    zstag = ph / g
    zsfc = zstag[0:1]
    zw = zstag - zsfc  # height above surface at w-levels
    nz = np.asarray(case.state.qke).shape[0]
    zw_mass = zw[:nz]
    ust_b = ust[None]
    qke_kts = 1.5 * np.maximum(ust_b, 0.02) ** 2 * (B1 * PMZ) ** (2.0 / 3.0)
    taper = np.maximum((ust_b * 700.0 - zw_mass) / (np.maximum(ust_b, 0.01) * 700.0), 0.01)
    qke_seed = np.maximum(qke_kts * taper, 1.0e-5)
    return jnp.asarray(qke_seed, dtype=case.state.qke.dtype), float(np.nanmax(ust)), float(np.nanmin(ust))


def main():
    t0 = time.perf_counter()
    case = build_l2_d02_replay_case(RUN_DIR, domain="d02", parent_domain="d01")
    qke_t0 = np.asarray(case.state.qke)
    seed, ust_max, ust_min = _wrf_coldstart_qke(case)
    seed_np = np.asarray(seed)

    baseline = _trace(case, None)
    seeded = _trace(case, seed)

    base_blowup = next((r["step"] for r in baseline if not r.get("all_fin", True)), None)
    seed_blowup = next((r["step"] for r in seeded if not r.get("all_fin", True)), None)

    out = {
        "schema": "V090D02ReplayQkeColdstartUnit",
        "platform": "cpu",
        "x64": True,
        "run_dir": RUN_DIR,
        "dt_s": DT,
        "steps": STEPS,
        "qke_t0_parent": {
            "min": float(qke_t0.min()),
            "max": float(qke_t0.max()),
            "all_zero": bool(np.all(qke_t0 == 0.0)),
            "shape": list(qke_t0.shape),
        },
        "surface_ustar_t0": {"min": ust_min, "max": ust_max},
        "wrf_coldstart_seed": {
            "formula": "qke(kts)=1.5*max(ust,0.02)^2*(b1*pmz)^(2/3); taper=max((ust*700-zw)/(max(ust,0.01)*700),0.01)",
            "wrf_ref": "phys/module_bl_mynnedmf.F:691 (driver) / :1331 (mym_initialize); b1=24, pmz=1, qkemin=1e-5",
            "min": float(seed_np.min()),
            "max": float(seed_np.max()),
            "mean": float(seed_np.mean()),
        },
        "baseline_qke0_trace": baseline,
        "wrf_seed_trace": seeded,
        "baseline_first_nonfinite_step": base_blowup,
        "seed_first_nonfinite_step": seed_blowup,
        "verdict": (
            "SEED CURES" if (base_blowup is not None and seed_blowup is None)
            else "BOTH FINITE (cpu, no dycore)" if (base_blowup is None and seed_blowup is None)
            else "INCONCLUSIVE"
        ),
        "elapsed_s": time.perf_counter() - t0,
    }
    outpath = Path(__file__).with_suffix(".json")
    outpath.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("WROTE", outpath)


if __name__ == "__main__":
    main()
