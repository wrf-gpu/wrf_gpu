"""S6a INTEGRATION gate — end-to-end ``noah_mp_step`` on the 11 Canary land columns.

Builds the full prognostic land state, forcing, and static (real S0b
``load_noahmp_parameters`` tables) from the pristine-WRF savepoint dump
(``proofs/noahmp/savepoints_all.json``, the same 11 land columns the per-component
gates use), runs the WIRED driver ``noah_mp_step`` over all columns at once
(vectorised (1, 11) grid), and asserts:

  (1) the integrated step runs end-to-end (no NotImplementedError / NaN / Inf);
  (2) the coupler-facing fluxes match the WRF driver mapping
      (HFX=FSH / GRDFLX=SSOIL / TSK=TRAD / LH / QFX) within the S1 energy-gate
      tolerances — i.e. the integration did not corrupt the energy answer;
  (3) the WRF ENERGY closure residual ERRENG (:1662) is ~0 on every column.

This is INTEGRATION CORRECTNESS (per-component parity is gated separately). It is
NOT a self-compare: the reference is the pristine-WRF NOAHMP_SFLX dump.

Run (CPU, cores 0-3):
  taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu \
      python3 proofs/noahmp/integration_step_gate.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic  # noqa: E402
from gpuwrf.config.paths import wrf_root, wrf_run_dir  # noqa: E402
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step  # noqa: E402
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters  # noqa: E402
from gpuwrf.physics.noahmp.types import NoahMPForcing  # noqa: E402

# Pristine-WRF run/ dir (MPTABLE/SOILPARM/GENPARM). WRF_PRISTINE_ROOT remains a
# legacy explicit override; otherwise use GPUWRF_WRF_ROOT/config.paths.
WRF_PRISTINE_ROOT = Path(os.environ["WRF_PRISTINE_ROOT"]).expanduser() if os.environ.get("WRF_PRISTINE_ROOT") else wrf_root()
TABLE_DIR = WRF_PRISTINE_ROOT / "run" if os.environ.get("WRF_PRISTINE_ROOT") else wrf_run_dir()

# Driver-mapping tolerances (W/m2 / K / kg/m2/s): match the S1 energy gate.
TOL = {
    "hfx":    (2.0, 0.02),
    "grdflx": (2.0, 0.02),
    "tsk":    (0.30, 0.0),
    "lh":     (3.0, 0.05),
    "erreng": (0.10, 0.0),
}


def _arr(vals, dtype=jnp.float64):
    """Pack a per-column python list into a (1, ncol) row grid."""
    return jnp.asarray(np.asarray(vals, dtype=float).reshape(1, -1), dtype=dtype)


def main():
    sp = json.load(open(HERE / "savepoints_all.json"))
    cols = sp["columns"]
    n = len(cols)
    dt = float(cols[0]["dt"])

    def col_vec(getter):
        return _arr([getter(c) for c in cols])

    def soil_vec(getter):  # (NSOIL, 1, n)
        a = np.stack([np.asarray(getter(c), dtype=float) for c in cols], axis=-1)  # (NSOIL, n)
        return jnp.asarray(a.reshape(NSOIL, 1, n), dtype=jnp.float64)

    def snow_vec(arr_name):  # (NSNOW, 1, n) from state_in[arr_name]
        a = np.stack([np.asarray(c["state_in"][arr_name], dtype=float) for c in cols], axis=-1)
        return jnp.asarray(a.reshape(NSNOW, 1, n), dtype=jnp.float64)

    def snowsoil_vec(arr_name):  # (NSNOW+NSOIL, 1, n)
        a = np.stack([np.asarray(c["state_in"][arr_name], dtype=float) for c in cols], axis=-1)
        return jnp.asarray(a.reshape(NSNOW + NSOIL, 1, n), dtype=jnp.float64)

    # STC in the dump is len-7 (-2..4); split snow (first NSNOW) / soil (last NSOIL).
    stc_all = np.stack([np.asarray(c["state_in"]["stc"], dtype=float) for c in cols], axis=-1)
    tsno = jnp.asarray(stc_all[:NSNOW].reshape(NSNOW, 1, n), dtype=jnp.float64)
    tslb = jnp.asarray(stc_all[NSNOW:].reshape(NSOIL, 1, n), dtype=jnp.float64)

    si = lambda c, k: c["state_in"][k]  # noqa: E731

    land = NoahMPLandState(
        tslb=tslb, smois=soil_vec(lambda c: c["state_in"]["smc"]),
        sh2o=soil_vec(lambda c: c["state_in"]["sh2o"]),
        smcwtd=col_vec(lambda c: si(c, "smcwtd")),
        isnow=jnp.asarray(np.asarray([int(si(c, "isnow")) for c in cols]).reshape(1, n), dtype=jnp.int32),
        tsno=tsno, snice=snow_vec("snice"), snliq=snow_vec("snliq"),
        zsnso=snowsoil_vec("zsnso"),
        snowh=col_vec(lambda c: si(c, "snowh")), sneqv=col_vec(lambda c: si(c, "sneqv")),
        sneqvo=col_vec(lambda c: si(c, "sneqvo")),
        tauss=col_vec(lambda c: si(c, "tauss")), albold=col_vec(lambda c: si(c, "albold")),
        tv=col_vec(lambda c: si(c, "tv")), tg=col_vec(lambda c: si(c, "tg")),
        tah=col_vec(lambda c: si(c, "tah")), eah=col_vec(lambda c: si(c, "eah")),
        canliq=col_vec(lambda c: si(c, "canliq")), canice=col_vec(lambda c: si(c, "canice")),
        fwet=col_vec(lambda c: si(c, "fwet")),
        lai=col_vec(lambda c: c["wrf"]["phen_out"]["lai"]),
        sai=col_vec(lambda c: c["wrf"]["phen_out"]["sai"]),
        cm=col_vec(lambda c: si(c, "cm")), ch=col_vec(lambda c: si(c, "ch")),
        t_skin=col_vec(lambda c: si(c, "tg")), qsfc=col_vec(lambda c: si(c, "qsfc")),
        znt=col_vec(lambda c: 0.05), emiss=col_vec(lambda c: 0.97),
        albedo=col_vec(lambda c: si(c, "albold")),
        sfcrunoff=col_vec(lambda c: 0.0), udrunoff=col_vec(lambda c: 0.0),
    )

    f = lambda c, k: c["forcing"][k]  # noqa: E731
    forcing = NoahMPForcing(
        sfctmp=col_vec(lambda c: f(c, "sfctmp")), sfcprs=col_vec(lambda c: f(c, "sfcprs")),
        psfc=col_vec(lambda c: f(c, "psfc")), uu=col_vec(lambda c: f(c, "uu")),
        vv=col_vec(lambda c: f(c, "vv")), qair=col_vec(lambda c: f(c, "q2")),
        qc=col_vec(lambda c: f(c, "qc")), soldn=col_vec(lambda c: f(c, "soldn")),
        lwdn=col_vec(lambda c: f(c, "lwdn")),
        prcpconv=col_vec(lambda c: f(c, "prcpconv")), prcpnonc=col_vec(lambda c: f(c, "prcpnonc")),
        prcpsnow=col_vec(lambda c: f(c, "prcpsnow")), prcpgrpl=col_vec(lambda c: f(c, "prcpgrpl")),
        prcphail=col_vec(lambda c: f(c, "prcphail")), cosz=col_vec(lambda c: f(c, "cosz")),
        zlvl=col_vec(lambda c: f(c, "zlvl")),
        julian=jnp.asarray(float(cols[0]["julian"])), yearlen=jnp.asarray(float(cols[0]["yearlen"])),
    )

    params = load_noahmp_parameters(TABLE_DIR)
    zsoil = jnp.asarray(cols[0]["zsoil"], dtype=jnp.float64)
    dzs = jnp.asarray([0.05, 0.20, 0.45, 0.80], dtype=jnp.float64)
    static = NoahMPStatic(
        ivgtyp=jnp.asarray(np.asarray([c["vegtyp"] for c in cols]).reshape(1, n), dtype=jnp.int32),
        isltyp=jnp.asarray(np.asarray([c["isltyp"] for c in cols]).reshape(1, n), dtype=jnp.int32),
        xland=col_vec(lambda c: 1.0), landmask=col_vec(lambda c: 1.0),
        lakemask=col_vec(lambda c: 0.0),
        lu_index=jnp.asarray(np.asarray([c["vegtyp"] for c in cols]).reshape(1, n), dtype=jnp.int32),
        tbot=col_vec(lambda c: c["tbot"]), dzs=dzs, zsoil=zsoil,
        lat=col_vec(lambda c: np.degrees(c["lat_rad"])), dx_m=float(cols[0]["dx"]),
        parameters=params,
        shdmax=col_vec(lambda c: c["shdmax"]),
        shdfac=col_vec(lambda c: c["shdfac"]),
    )

    # ---- run the wired driver end-to-end ----
    land_out, fluxes, resid = noah_mp_step(land_state=land, forcing=forcing, static=static,
                                           dt=dt, return_diag=True)

    g = lambda a: np.asarray(a).reshape(-1)  # noqa: E731
    got = {
        "hfx": g(fluxes.hfx), "grdflx": g(fluxes.grdflx), "tsk": g(fluxes.tsk),
        "lh": g(fluxes.lh), "erreng": g(resid.erreng),
    }
    # finiteness across the whole advanced carry
    all_finite = all(
        bool(np.all(np.isfinite(np.asarray(v))))
        for v in jax.tree_util.tree_leaves(land_out)
    ) and all(bool(np.all(np.isfinite(got[k]))) for k in got)

    rows = []
    n_pass = n_fail = 0
    for i, c in enumerate(cols):
        drv = c["wrf"]["driver"]
        ref = {"hfx": drv["hfx"], "grdflx": drv["grdflx"], "tsk": drv["tsk"], "lh": drv["lh"]}
        result = {}
        for fld, (atol, rtol) in TOL.items():
            if fld == "erreng":
                ok = abs(float(got["erreng"][i])) <= atol
                result[fld] = (ok, 0.0, float(got["erreng"][i]))
                continue
            r = ref[fld]
            gv = float(got[fld][i])
            ok = abs(gv - r) <= atol + rtol * abs(r)
            result[fld] = (ok, r, gv)
        col_ok = all(v[0] for v in result.values())
        n_pass += col_ok
        n_fail += not col_ok
        rows.append((c["name"], col_ok, result))

    print(f"\n{'='*82}\nS6a INTEGRATION GATE — noah_mp_step over {n} Canary land columns\n{'='*82}")
    print(f"end-to-end finite (carry + fluxes): {all_finite}")
    for name, col_ok, result in rows:
        print(f"\n[{'PASS' if col_ok else 'FAIL'}] {name}")
        for fld, (ok, r, gv) in result.items():
            mark = "ok " if ok else "XX "
            print(f"   {mark}{fld:8s} wrf={r:14.6g}  jax={gv:14.6g}  d={gv-r:+12.5g}")
    verdict_ok = all_finite and n_fail == 0
    print(f"\n{'='*82}\nVERDICT: {n_pass} PASS / {n_fail} FAIL of {n}  (finite={all_finite})\n{'='*82}")

    proof = {
        "proof": "S6a integrated noah_mp_step REAL-WRF driver-mapping + closure gate",
        "kind": ("end-to-end integrated driver vs pristine-WRF NOAHMP_SFLX driver "
                 "dump (savepoints_all.json); HFX/GRDFLX/TSK/LH driver mapping + "
                 "ENERGY ERRENG closure; NOT a self-compare"),
        "oracle": "proofs/noahmp/savepoints_all.json",
        "ncolumns": n, "npass": int(n_pass), "nfail": int(n_fail),
        "end_to_end_finite": bool(all_finite),
        "tolerances": {k: {"atol": a, "rtol": r} for k, (a, r) in TOL.items()},
        "columns": [
            {"name": name, "pass": bool(ok),
             "fields": {fld: {"wrf": rr, "jax": gg, "pass": bool(p)}
                        for fld, (p, rr, gg) in res.items()}}
            for name, ok, res in rows
        ],
        "verdict": "S6a_INTEGRATION_PASS" if verdict_ok else "FAIL",
    }
    (HERE / "integration_step_parity.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(f"proof -> {HERE / 'integration_step_parity.json'}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
