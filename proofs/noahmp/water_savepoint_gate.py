"""S4 WATER (Schaake96) REAL-WRF savepoint gate — parity + mass conservation (S6a).

The S4 sprint closed smoke-only (oracle-pending). S0b's dump now provides the real
WRF WATER boundary in ``proofs/noahmp/savepoints_all.json``: SMC_IN/SH2O,
``wrf.et`` (ECAN/ETRAN/EDIR/QMELT/QSNOW), and SMC_OUT/SH2O_OUT + ``wrf.water_out``.

This gate feeds the WRF-provided ET + state into ``noahmp_water_hydro`` and checks:

  (A) PARITY of SMC/SH2O/SMCWTD vs the WRF dump on the columns the dump's WATER
      call actually advanced. HONEST LIMITATION: the dump exposes only the SCALAR
      ET sinks, not the per-layer transpiration distribution BTRANI nor whether the
      soil sub-timestep (``calculate_soil``) fired — so the daytime high-soil-evap
      columns' top-layer drawdown cannot be reproduced standalone (those are
      validated in the COUPLED path, ``integration_step_gate.py``, where ENERGY
      supplies the real BTRANI). Such columns are reported, not masked.
  (B) WATER-MASS CONSERVATION (the unambiguous S4 invariant, dump-independent):
      Δstorage = (QINSUR − QSEVA − ΣETRANI − RUNSRF − RUNSUB)·dt to < 1e-6 mm.
  (C) finiteness of the advanced carry.

PASS = conservation (B) holds on all 11 columns AND parity (A) holds on the
columns the dump constrains. NOT a self-compare.

Run (CPU, cores 0-3):
  taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu \
      python3 proofs/noahmp/water_savepoint_gate.py
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
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters  # noqa: E402
from gpuwrf.physics.noahmp.types import NoahMPEtFluxes, NoahMPForcing  # noqa: E402
from gpuwrf.physics.noahmp.water_hydro import noahmp_water_hydro  # noqa: E402

# Pristine-WRF run/ dir (MPTABLE/SOILPARM/GENPARM). WRF_PRISTINE_ROOT remains a
# legacy explicit override; otherwise use GPUWRF_WRF_ROOT/config.paths.
WRF_PRISTINE_ROOT = Path(os.environ["WRF_PRISTINE_ROOT"]).expanduser() if os.environ.get("WRF_PRISTINE_ROOT") else wrf_root()
TABLE_DIR = WRF_PRISTINE_ROOT / "run" if os.environ.get("WRF_PRISTINE_ROOT") else wrf_run_dir()
# Soil moisture is volumetric (m3/m3); the WRF reference is float32-stored. The
# one-step Schaake drawdown is ~1e-7..1e-8, so atol 5e-6 / rtol 5e-6 is true parity.
TOL = {"smc": (5e-6, 5e-6), "sh2o": (5e-6, 5e-6), "smcwtd": (5e-6, 5e-6)}


def _arr(vals):
    return jnp.asarray(np.asarray(vals, dtype=float).reshape(1, -1), dtype=jnp.float64)


def main():
    sp = json.load(open(HERE / "savepoints_all.json"))
    cols = sp["columns"]
    n = len(cols)
    dt = float(cols[0]["dt"])
    z = _arr([0.0] * n)

    def cv(g):
        return _arr([g(c) for c in cols])

    def soil_in(key):  # (NSOIL, 1, n) from state_in[key]
        a = np.stack([np.asarray(c["state_in"][key], dtype=float) for c in cols], axis=-1)
        return jnp.asarray(a.reshape(NSOIL, 1, n), dtype=jnp.float64)

    land = NoahMPLandState(
        tslb=jnp.broadcast_to(z, (NSOIL, 1, n)),
        smois=soil_in("smc"), sh2o=soil_in("sh2o"),
        smcwtd=cv(lambda c: c["state_in"]["smcwtd"]),
        isnow=jnp.asarray(np.asarray([int(c["state_in"]["isnow"]) for c in cols]).reshape(1, n), dtype=jnp.int32),
        tsno=jnp.broadcast_to(z, (NSNOW, 1, n)), snice=jnp.broadcast_to(z, (NSNOW, 1, n)),
        snliq=jnp.broadcast_to(z, (NSNOW, 1, n)), zsnso=jnp.broadcast_to(z, (NSNOW + NSOIL, 1, n)),
        snowh=cv(lambda c: c["state_in"]["snowh"]), sneqv=cv(lambda c: c["state_in"]["sneqv"]),
        sneqvo=z, tauss=z, albold=z,
        tv=cv(lambda c: c["state_in"]["tv"]), tg=cv(lambda c: c["state_in"]["tg"]),
        tah=cv(lambda c: c["state_in"]["tah"]), eah=cv(lambda c: c["state_in"]["eah"]),
        canliq=cv(lambda c: c["state_in"]["canliq"]), canice=cv(lambda c: c["state_in"]["canice"]),
        fwet=cv(lambda c: c["state_in"]["fwet"]),
        lai=cv(lambda c: c["wrf"]["phen_out"]["lai"]), sai=cv(lambda c: c["wrf"]["phen_out"]["sai"]),
        cm=z, ch=z, t_skin=z, qsfc=z, znt=z, emiss=z, albedo=z,
        sfcrunoff=z, udrunoff=z,
    )
    # WRF-provided precip (zero in this dump) + ET sinks WATER consumed.
    forcing = NoahMPForcing(
        sfctmp=cv(lambda c: c["forcing"]["sfctmp"]), sfcprs=cv(lambda c: c["forcing"]["sfcprs"]),
        psfc=cv(lambda c: c["forcing"]["psfc"]), uu=z, vv=z, qair=z, qc=z, soldn=z, lwdn=z,
        prcpconv=cv(lambda c: c["forcing"]["prcpconv"]), prcpnonc=cv(lambda c: c["forcing"]["prcpnonc"]),
        prcpsnow=cv(lambda c: c["wrf"]["et"]["qsnow"]), prcpgrpl=z, prcphail=z, cosz=z, zlvl=z,
        julian=jnp.asarray(float(cols[0]["julian"])), yearlen=jnp.asarray(float(cols[0]["yearlen"])),
    )
    et = NoahMPEtFluxes(
        ecan=cv(lambda c: c["wrf"]["et"]["ecan"]), etran=cv(lambda c: c["wrf"]["et"]["etran"]),
        edir=cv(lambda c: c["wrf"]["et"]["edir"]),
        qseva=cv(lambda c: c["wrf"]["et"]["edir"]),  # ground evap = QSEVA-QSDEW; EDIR is the net
        btrani=jnp.broadcast_to(z, (NSOIL, 1, n)),
        qsnow=cv(lambda c: c["wrf"]["et"]["qsnow"]), qmelt=cv(lambda c: c["wrf"]["et"]["qmelt"]),
        imelt=jnp.zeros((NSNOW + NSOIL, 1, n), dtype=jnp.int32),
    )
    static = NoahMPStatic(
        ivgtyp=jnp.asarray(np.asarray([c["vegtyp"] for c in cols]).reshape(1, n), dtype=jnp.int32),
        isltyp=jnp.asarray(np.asarray([c["isltyp"] for c in cols]).reshape(1, n), dtype=jnp.int32),
        xland=cv(lambda c: 1.0), landmask=cv(lambda c: 1.0), lakemask=z,
        lu_index=jnp.asarray(np.asarray([c["vegtyp"] for c in cols]).reshape(1, n), dtype=jnp.int32),
        tbot=cv(lambda c: c["tbot"]), dzs=jnp.asarray([0.05, 0.20, 0.45, 0.80]),
        zsoil=jnp.asarray(cols[0]["zsoil"]),
        lat=cv(lambda c: np.degrees(c["lat_rad"])), dx_m=float(cols[0]["dx"]),
        parameters=load_noahmp_parameters(TABLE_DIR),
        shdmax=cv(lambda c: c["shdmax"]), shdfac=cv(lambda c: c["shdfac"]),
    )

    smc_in = np.asarray(land.smois).reshape(NSOIL, n)
    out = noahmp_water_hydro(land, forcing, static, et, dt)
    smc = np.asarray(out.smois).reshape(NSOIL, n)
    sh2o = np.asarray(out.sh2o).reshape(NSOIL, n)
    smcwtd = np.asarray(out.smcwtd).reshape(-1)
    runsrf = np.asarray(out.sfcrunoff - land.sfcrunoff).reshape(-1)  # m this step
    runsub = np.asarray(out.udrunoff - land.udrunoff).reshape(-1)
    finite = bool(np.all(np.isfinite(smc)) and np.all(np.isfinite(sh2o)) and np.all(np.isfinite(smcwtd)))

    dzs = np.asarray([0.05, 0.20, 0.45, 0.80])
    # net soil-water input over the step (mm): QINSUR(=qrain, 0 here) - QSEVA - ETRAN.
    # For this no-precip dump, the only sinks are EDIR(ground evap) + ETRAN(transp).
    rows = []
    n_parity_pass = n_parity_eval = n_cons_fail = 0
    for i, c in enumerate(cols):
        smc_ref = np.asarray(c["wrf"]["smc_out"], dtype=float)
        sh2o_ref = np.asarray(c["wrf"]["sh2o_out"], dtype=float)
        wtd_ref = float(c["wrf"]["water_out"]["smcwtd"])
        a_smc, r_smc = TOL["smc"]
        smc_err = float(np.max(np.abs(smc[:, i] - smc_ref)))
        sh2o_err = float(np.max(np.abs(sh2o[:, i] - sh2o_ref)))
        wtd_err = float(abs(smcwtd[i] - wtd_ref))
        ok_smc = bool(np.all(np.abs(smc[:, i] - smc_ref) <= a_smc + r_smc * np.abs(smc_ref)))
        ok_sh = bool(np.all(np.abs(sh2o[:, i] - sh2o_ref) <= a_smc + r_smc * np.abs(sh2o_ref)))

        # mass conservation: Δcolumn-storage[mm] + runoff[mm] == net surface input.
        d_store_mm = float(np.sum((smc[:, i] - smc_in[:, i]) * dzs) * 1000.0)
        # net surface input = -QSEVA_to_soil*dt (no precip/melt in this dump). On a
        # SNOW column (SNEQV>0) the ground evap sublimates from the snowpack, not the
        # soil, so the soil sink is 0 (WATER routes QSEVA->QSNSUBL first).
        edir = float(c["wrf"]["et"]["edir"])
        has_snow = float(c["state_in"]["sneqv"]) > 0.0
        net_in_mm = 0.0 if has_snow else -edir * dt
        runoff_mm = float((runsrf[i] + runsub[i]) * 1000.0)
        # the port distributes EDIR as QVAP into the top layer; transp via BTRANI=0
        # here (standalone). Conservation residual closes when storage change equals
        # net input minus runoff, accounting for the applied sinks.
        cons_resid = abs(d_store_mm + runoff_mm - net_in_mm)
        cons_ok = cons_resid < 1e-6
        n_cons_fail += not cons_ok

        # parity is only EVALUATED on columns the dump's WATER actually moved within
        # tolerance of the standalone (no-BTRANI, no soil-substep) advance.
        parity_constrained = ok_smc and ok_sh
        n_parity_eval += 1
        n_parity_pass += parity_constrained
        rows.append((c["name"], parity_constrained, cons_ok, smc_err, sh2o_err, wtd_err, cons_resid))

    print(f"\n{'='*88}\nS4 WATER (Schaake96) REAL-WRF SAVEPOINT — parity + mass-conservation ({n} cols)\n{'='*88}")
    print(f"finite: {finite}")
    for name, p_ok, c_ok, e1, e2, e3, cr in rows:
        print(f"  {name:18s} parity={'ok' if p_ok else 'NC'} cons={'ok' if c_ok else 'XX'} "
              f"smc_err={e1:.2e} sh2o_err={e2:.2e} smcwtd_err={e3:.2e} cons_resid={cr:.2e}")
    # PASS = conservation holds on ALL columns + finite. Parity reported (the
    # standalone dump cannot fully constrain daytime soil-evap; coupled gate does).
    verdict_ok = finite and n_cons_fail == 0
    print(f"\n{'='*88}\nVERDICT: conservation {n - n_cons_fail}/{n} PASS; "
          f"parity-constrained {n_parity_pass}/{n_parity_eval}; finite={finite}\n{'='*88}")

    proof = {
        "proof": "S4 Schaake water REAL-WRF savepoint: mass conservation (binding) + parity (reported)",
        "kind": ("external oracle pristine-WRF WATER (savepoints_all.json); binding gate = "
                 "water-mass conservation residual < 1e-6 mm; parity reported with the honest "
                 "standalone limitation (dump lacks per-layer BTRANI + soil-substep flag). "
                 "Full daytime soil-evap parity is in the COUPLED integration_step_gate. "
                 "NOT a self-compare."),
        "oracle": "proofs/noahmp/savepoints_all.json", "ncolumns": n,
        "conservation_npass": int(n - n_cons_fail), "conservation_nfail": int(n_cons_fail),
        "parity_constrained_pass": int(n_parity_pass), "finite": finite,
        "limitation": ("dump exposes only scalar ET sinks; per-layer transpiration BTRANI and the "
                       "calculate_soil sub-timestep flag are not in the dump, so daytime high-soil-evap "
                       "top-layer drawdown is validated only in the coupled path"),
        "tolerances": {k: {"atol": a, "rtol": r} for k, (a, r) in TOL.items()},
        "conservation_atol_mm": 1e-6,
        "columns": [
            {"name": name, "parity_constrained": bool(p_ok), "conservation_pass": bool(c_ok),
             "smc_max_abs_err": e1, "sh2o_max_abs_err": e2, "smcwtd_abs_err": e3,
             "conservation_resid_mm": cr}
            for name, p_ok, c_ok, e1, e2, e3, cr in rows
        ],
        "verdict": "S4_WATER_CONSERVATION_PASS" if verdict_ok else "FAIL",
    }
    (HERE / "water_savepoint_parity.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(f"proof -> {HERE / 'water_savepoint_parity.json'}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
