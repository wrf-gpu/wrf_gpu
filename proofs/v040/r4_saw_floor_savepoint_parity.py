#!/usr/bin/env python
"""Round-4 MYNN s_aw momentum-Km floor savepoint parity.

Compares the fixed JAX U/V implicit solve against the unmodified WRF
``module_bl_mynnedmf`` oracle on the real d03 convective column from
``proofs/mynn_edmf``. The WRF oracle is produced by
``proofs/mynn_edmf/fortran_oracle/oracle.f90``, which links pristine WRF objects
and calls real ``DMP_mf`` + ``mynn_tendencies``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics.mynn_edmf import XLVCP, dmp_mf_columns  # noqa: E402
from gpuwrf.physics.mynn_pbl import (  # noqa: E402
    MynnPBLColumnState,
    _apply_s_aw_stability_floor,
    _diffusion_solve_with_surface,
    _rho_interfaces,
)

COL = ROOT / "proofs/mynn_edmf/column_d03_12z.json"
FORT = ROOT / "proofs/mynn_edmf/fortran_oracle/oracle_out.txt"
OUT = ROOT / "proofs/v040/r4_saw_floor_savepoint_parity.json"

TOL = {
    "s_aw_rel_max": 0.05,       # existing DMP_mf WRF-r4 vs JAX-fp64 oracle tol
    "kmdz_abs": 5.0e-7,         # WRF r4 machine/format floor for K*dz arrays
    "rubvblten_abs": 5.0e-7,    # WRF r4 machine/format floor for Du/Dv tendencies
}


def parse_fort(path: Path) -> dict[str, np.ndarray | float]:
    out: dict[str, np.ndarray | float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if "," in val:
            out[key] = np.array([float(x) for x in val.split(",")], dtype=np.float64)
        else:
            out[key] = float(val)
    return out


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1.0e-30))


def maxabs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def build_state(c: dict) -> MynnPBLColumnState:
    pr = c["profiles"]
    arr = lambda name: jnp.array(pr[name], dtype=jnp.float64)[None, :]
    z = jnp.zeros_like(arr("th"))
    return MynnPBLColumnState(
        arr("u"), arr("v"), arr("w"), arr("th"), arr("qv"), 0.5 * arr("qke"),
        arr("p"), arr("rho"), arr("dz"), z, z, z,
    )


def jax_saw(c: dict) -> np.ndarray:
    pr = c["profiles"]
    su = c["surface"]
    arr = lambda name: jnp.array(pr[name], dtype=jnp.float64)[None, :]
    qv = arr("qv")
    sqv = qv / (1.0 + qv)
    sqc = arr("qc") / (1.0 + qv)
    sqw = sqv + sqc + arr("qi") / (1.0 + qv)
    th = arr("th")
    thl = th - XLVCP / arr("exner") * sqc
    thv = th * (1.0 + 0.608 * sqv)
    zw = jnp.concatenate([jnp.zeros((1, 1)), jnp.cumsum(arr("dz"), axis=-1)], axis=-1)
    s1 = lambda x: jnp.array([x], dtype=jnp.float64)
    mf = dmp_mf_columns(
        sqw, sqv, sqc, arr("u"), arr("v"), arr("w"), th, thl, thv, arr("tk"), arr("qke"),
        arr("p"), arr("exner"), arr("rho"), arr("dz"), zw,
        ust=s1(su["ust"]), flt=s1(su["flt"]), fltv=s1(su["fltv"]),
        flq=s1(su["flq"]), flqv=s1(su["flqv"]),
        pblh=s1(su["pblh"]), ts=s1(su["tsk"]), dx=su["dx"],
        xland=s1(su["xland"]), dt=c["config"]["delt"],
    )
    return np.asarray(mf["s_aw"][0], dtype=np.float64)


def main() -> int:
    c = json.loads(COL.read_text(encoding="utf-8"))
    fo = parse_fort(FORT)
    required = ("edmf_s_aw", "edmf_dfm", "edmf_kmdz_eff", "edmf_Du", "edmf_Dv")
    missing = [k for k in required if k not in fo]
    if missing:
        raise SystemExit(f"missing WRF oracle keys {missing}; rerun proofs/mynn_edmf/fortran_oracle/build_and_run.sh")

    st = build_state(c)
    dt = float(c["config"]["delt"])
    su = c["surface"]
    dfm = jnp.array(np.asarray(fo["edmf_dfm"], dtype=np.float64)[None, :])
    wrf_saw = np.asarray(fo["edmf_s_aw"], dtype=np.float64)
    saw = jnp.array(wrf_saw[None, :])

    kmdz_base = _rho_interfaces(st, dfm)
    kmdz_fixed = _apply_s_aw_stability_floor(kmdz_base, saw)
    kmdz_nofloor = np.asarray(kmdz_base[0], dtype=np.float64)
    kmdz_jax = np.asarray(kmdz_fixed[0], dtype=np.float64)
    kmdz_wrf = np.asarray(fo["edmf_kmdz_eff"], dtype=np.float64)

    rhosfc = float(su["psfc"]) / (287.0 * (float(c["profiles"]["tk"][0]) + 0.608 * float(c["profiles"]["qv"][0])))
    bottom_drag = jnp.array([rhosfc * float(su["ust"]) ** 2 / float(su["wspd"])], dtype=jnp.float64)
    zeros = jnp.zeros((1,), dtype=jnp.float64)
    u_fixed = _diffusion_solve_with_surface(st.u, dfm, st, dt, zeros, bottom_drag, s_aw_floor=saw)
    v_fixed = _diffusion_solve_with_surface(st.v, dfm, st, dt, zeros, bottom_drag, s_aw_floor=saw)
    u_nofloor = _diffusion_solve_with_surface(st.u, dfm, st, dt, zeros, bottom_drag)
    v_nofloor = _diffusion_solve_with_surface(st.v, dfm, st, dt, zeros, bottom_drag)
    du_jax = (np.asarray(u_fixed[0], dtype=np.float64) - np.asarray(st.u[0], dtype=np.float64)) / dt
    dv_jax = (np.asarray(v_fixed[0], dtype=np.float64) - np.asarray(st.v[0], dtype=np.float64)) / dt
    du_nofloor = (np.asarray(u_nofloor[0], dtype=np.float64) - np.asarray(st.u[0], dtype=np.float64)) / dt
    dv_nofloor = (np.asarray(v_nofloor[0], dtype=np.float64) - np.asarray(st.v[0], dtype=np.float64)) / dt
    du_wrf = np.asarray(fo["edmf_Du"], dtype=np.float64)
    dv_wrf = np.asarray(fo["edmf_Dv"], dtype=np.float64)

    saw_jax = jax_saw(c)[: len(wrf_saw)]
    floor_lift = kmdz_jax - kmdz_nofloor
    active_floor_levels = np.where(np.abs(floor_lift) > 0.0)[0].tolist()
    nofloor_du_resid = maxabs(du_nofloor, du_wrf)
    nofloor_dv_resid = maxabs(dv_nofloor, dv_wrf)
    fixed_du_resid = maxabs(du_jax, du_wrf)
    fixed_dv_resid = maxabs(dv_jax, dv_wrf)

    checks = {
        "s_aw_verified_against_wrf": {
            "max_rel_error": relerr(saw_jax, wrf_saw),
            "tolerance": TOL["s_aw_rel_max"],
            "pass": relerr(saw_jax, wrf_saw) <= TOL["s_aw_rel_max"],
        },
        "kmdz_eff_fixed_vs_wrf": {
            "max_abs": maxabs(kmdz_jax, kmdz_wrf),
            "rmse": rmse(kmdz_jax, kmdz_wrf),
            "tolerance_abs": TOL["kmdz_abs"],
            "pass": maxabs(kmdz_jax, kmdz_wrf) <= TOL["kmdz_abs"],
        },
        "rubvblten_fixed_vs_wrf": {
            "Du_max_abs": fixed_du_resid,
            "Dv_max_abs": fixed_dv_resid,
            "Du_rmse": rmse(du_jax, du_wrf),
            "Dv_rmse": rmse(dv_jax, dv_wrf),
            "tolerance_abs": TOL["rubvblten_abs"],
            "pass": fixed_du_resid <= TOL["rubvblten_abs"] and fixed_dv_resid <= TOL["rubvblten_abs"],
        },
    }
    required_checks = ("s_aw_verified_against_wrf", "kmdz_eff_fixed_vs_wrf", "rubvblten_fixed_vs_wrf")
    floor_active = bool(active_floor_levels)
    nofloor_diagnostic = {
        "floor_active_on_this_savepoint": floor_active,
        "Du_nofloor_max_abs": nofloor_du_resid,
        "Dv_nofloor_max_abs": nofloor_dv_resid,
        "Du_fixed_improvement_factor": nofloor_du_resid / max(fixed_du_resid, 1.0e-30),
        "Dv_fixed_improvement_factor": nofloor_dv_resid / max(fixed_dv_resid, 1.0e-30),
        "interpretation": (
            "On this real d03 column, local WRF/JAX kmdz is already above "
            "0.5*s_aw on all interfaces, so the no-floor and fixed U/V tendencies "
            "are indistinguishable at WRF-r4 precision. This savepoint gates parity; "
            "the 2-date forecast gate measures operational impact."
        ),
    }
    result = {
        "schema": "v0.4.0-r4-saw-floor-savepoint-parity-2026-06-03",
        "verdict": "PASS" if all(checks[k]["pass"] for k in required_checks) else "FAIL",
        "oracle": {
            "type": "unmodified WRF module_bl_mynnedmf object-linked Fortran oracle",
            "source": str(FORT),
            "wrf_source_lines": {
                "kmdz_floor": "$WRF_PRISTINE_ROOT/phys/module_bl_mynnedmf.F:3990-3997",
                "uv_onoff_mf_terms": "$WRF_PRISTINE_ROOT/phys/module_bl_mynnedmf.F:3949-3956,4009-4034,4070-4095",
                "du_dv_outputs": "$WRF_PRISTINE_ROOT/phys/module_bl_mynnedmf.F:4055-4062,4116-4123",
            },
        },
        "predeclared_tolerances": TOL,
        "required_checks": list(required_checks),
        "column": c["meta"],
        "surface": {k: su[k] for k in ("ust", "wspd", "pblh", "fltv", "dx", "xland")},
        "checks": checks,
        "nofloor_diagnostic": nofloor_diagnostic,
        "floor_effect": {
            "active_interface_indices_0based": active_floor_levels,
            "max_kmdz_lift": float(np.max(floor_lift)),
            "kmdz_nofloor_first6": kmdz_nofloor[:6].tolist(),
            "kmdz_fixed_first6": kmdz_jax[:6].tolist(),
            "wrf_s_aw_first6": wrf_saw[:6].tolist(),
            "du_wrf_first6": du_wrf[:6].tolist(),
            "du_fixed_first6": du_jax[:6].tolist(),
            "du_nofloor_first6": du_nofloor[:6].tolist(),
            "dv_wrf_first6": dv_wrf[:6].tolist(),
            "dv_fixed_first6": dv_jax[:6].tolist(),
            "dv_nofloor_first6": dv_nofloor[:6].tolist(),
        },
        "interpretation": (
            "REAL omission from source reading: WRF applies the s_aw stability floor "
            "to kmdz even with bl_mynn_edmf_mom=0. This proof gates the fixed JAX "
            "formula against the WRF oracle; the paired no-floor diagnostic records "
            "whether this particular real column activates the floor."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in checks.items()}, indent=2, sort_keys=True))
    print("verdict:", result["verdict"])
    print("wrote", OUT)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
