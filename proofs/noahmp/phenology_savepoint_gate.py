"""S5 PHENOLOGY REAL-WRF savepoint parity gate (v0.2.0 S6a).

Feeds each pristine-WRF land column from ``proofs/noahmp/savepoints_all.json`` (the
S0b NOAHMP_SFLX dump, 11 Canary land columns with the true SHDMAX/SHDFAC) through
the JAX ``noahmp_phenology_table`` and asserts field-wise parity of LAI/SAI/FVEG
against ``wrf.phen_out`` (the WRF PHENOLOGY + FVEG-block outputs). NOT a
self-compare: the reference is the Fortran dump; the JAX port reads the SAME
MPTABLE via the S0b loader and the SAME column state.

This confirms the dveg=4 FVEG arbiter (module_sf_noahmplsm.F:864): FVEG = SHDMAX
(annual-max veg fraction), NOT SHDFAC — every column's wrf.phen_out.fveg equals
its shdmax, which the port now sources from NoahMPStatic.shdmax.

Run (CPU, cores 0-3):
  taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu \
      python3 proofs/noahmp/phenology_savepoint_gate.py
"""
from __future__ import annotations

import json
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
from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table  # noqa: E402
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters  # noqa: E402
from gpuwrf.physics.noahmp.types import NoahMPForcing  # noqa: E402

TABLE_DIR = Path("/home/enric/src/wrf_pristine/WRF/run")
# Tolerances at the savepoint's float32 storage precision (the WRF reference is
# dumped as float32, ~1e-7 relative): atol 1e-5, rtol 1e-5 is true WRF parity.
TOL = {"lai": (1e-5, 1e-5), "sai": (1e-5, 1e-5), "fveg": (1e-5, 1e-5)}


def _arr(vals):
    return jnp.asarray(np.asarray(vals, dtype=float).reshape(1, -1), dtype=jnp.float64)


def main():
    sp = json.load(open(HERE / "savepoints_all.json"))
    cols = sp["columns"]
    n = len(cols)
    z = _arr([0.0] * n)

    def cv(g):
        return _arr([g(c) for c in cols])

    land = NoahMPLandState(
        tslb=jnp.broadcast_to(z, (NSOIL, 1, n)), smois=jnp.broadcast_to(z, (NSOIL, 1, n)),
        sh2o=jnp.broadcast_to(z, (NSOIL, 1, n)), smcwtd=z,
        isnow=jnp.zeros((1, n), dtype=jnp.int32),
        tsno=jnp.broadcast_to(z, (NSNOW, 1, n)), snice=jnp.broadcast_to(z, (NSNOW, 1, n)),
        snliq=jnp.broadcast_to(z, (NSNOW, 1, n)), zsnso=jnp.broadcast_to(z, (NSNOW + NSOIL, 1, n)),
        snowh=cv(lambda c: c["state_in"]["snowh"]), sneqv=z, sneqvo=z, tauss=z, albold=z,
        tv=cv(lambda c: c["state_in"]["tv"]), tg=z, tah=z, eah=z,
        canliq=z, canice=z, fwet=z, lai=z, sai=z, cm=z, ch=z,
        t_skin=z, qsfc=z, znt=z, emiss=z, albedo=z, sfcrunoff=z, udrunoff=z,
    )
    forcing = NoahMPForcing(
        sfctmp=z, sfcprs=z, psfc=z, uu=z, vv=z, qair=z, qc=z, soldn=z, lwdn=z,
        prcpconv=z, prcpnonc=z, prcpsnow=z, prcpgrpl=z, prcphail=z, cosz=z, zlvl=z,
        julian=jnp.asarray(float(cols[0]["julian"])), yearlen=jnp.asarray(float(cols[0]["yearlen"])),
    )
    static = NoahMPStatic(
        ivgtyp=jnp.asarray(np.asarray([c["vegtyp"] for c in cols]).reshape(1, n), dtype=jnp.int32),
        isltyp=jnp.asarray(np.asarray([c["isltyp"] for c in cols]).reshape(1, n), dtype=jnp.int32),
        xland=cv(lambda c: 1.0), landmask=cv(lambda c: 1.0), lakemask=z,
        lu_index=jnp.asarray(np.asarray([c["vegtyp"] for c in cols]).reshape(1, n), dtype=jnp.int32),
        tbot=z, dzs=jnp.zeros(4), zsoil=jnp.zeros(4),
        lat=cv(lambda c: np.degrees(c["lat_rad"])), dx_m=1000.0,
        parameters=load_noahmp_parameters(TABLE_DIR),
        shdmax=cv(lambda c: c["shdmax"]), shdfac=cv(lambda c: c["shdfac"]),
    )

    phen = noahmp_phenology_table(land, forcing, static)
    g = lambda a: np.asarray(a).reshape(-1)  # noqa: E731
    got = {"lai": g(phen.lai), "sai": g(phen.sai), "fveg": g(phen.fveg)}

    rows = []
    n_pass = n_fail = 0
    for i, c in enumerate(cols):
        ref = c["wrf"]["phen_out"]
        result = {}
        for fld, (atol, rtol) in TOL.items():
            r = ref[fld]
            gv = float(got[fld][i])
            ok = abs(gv - r) <= atol + rtol * abs(r)
            result[fld] = (ok, r, gv)
        col_ok = all(v[0] for v in result.values())
        n_pass += col_ok
        n_fail += not col_ok
        rows.append((c["name"], col_ok, result))

    print(f"\n{'='*78}\nS5 PHENOLOGY REAL-WRF SAVEPOINT PARITY  ({n} columns)\n{'='*78}")
    for name, col_ok, result in rows:
        print(f"\n[{'PASS' if col_ok else 'FAIL'}] {name}")
        for fld, (ok, r, gv) in result.items():
            print(f"   {'ok ' if ok else 'XX '}{fld:5s} wrf={r:12.6g}  jax={gv:12.6g}  d={gv-r:+12.5g}")
    print(f"\n{'='*78}\nVERDICT: {n_pass} PASS / {n_fail} FAIL of {n}\n{'='*78}")

    proof = {
        "proof": "S5 phenology REAL-WRF savepoint parity (LAI/SAI/FVEG; dveg=4 FVEG=SHDMAX)",
        "kind": "external oracle: pristine-WRF NOAHMP_SFLX phen_out vs JAX port; NOT a self-compare",
        "oracle": "proofs/noahmp/savepoints_all.json", "ncolumns": n,
        "npass": int(n_pass), "nfail": int(n_fail),
        "fveg_arbiter": "module_sf_noahmplsm.F:864 dveg=4 FVEG=SHDMAX (verified: phen_out.fveg==shdmax)",
        "columns": [
            {"name": name, "pass": bool(ok),
             "fields": {f: {"wrf": rr, "jax": gg, "pass": bool(p)} for f, (p, rr, gg) in res.items()}}
            for name, ok, res in rows
        ],
        "verdict": "S5_PHENOLOGY_WRF_PARITY_PASS" if n_fail == 0 else "FAIL",
    }
    (HERE / "phenology_savepoint_parity.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(f"proof -> {HERE / 'phenology_savepoint_parity.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
