"""Noah-MP 2-m LSM diagnostic (T2MV / T2MB / combined T2) REAL-WRF parity gate.

THE faithful resolution of the operational land-T2: over LAND real WRF OVERWRITES
the surface-layer (MYNN/sfclay) 2-m temperature with the Noah-MP LSM diagnostic
``T2 = FVEG*T2MV + (1-FVEG)*T2MB`` (module_surface_driver.F:3469-3473), where T2MV is
the vegetated-tile 2-m air temperature (module_sf_noahmplsm.F:4148-4163) and T2MB the
bare-ground 2-m air temperature (:4461-4474). This gate feeds each pristine-WRF Noah-MP
savepoint column through the JAX port and asserts FIELD-WISE parity of T2MV/T2MB/T2
against the WRF-computed values.

EXTERNAL ORACLE — NOT a self-compare: the WRF reference T2MV/T2MB are emitted by the
compiled pristine ``module_sf_noahmplsm.o`` NOAHMP_SFLX driver
(proofs/noahmp/noahmp_offline_driver.F90, ``T2DIAG`` line) on real Canary d03 land
columns; the JAX port reads the IDENTICAL column input + the IDENTICAL WRF parameter
tables (via proofs/noahmp/energy_savepoint_gate.build_params/build_state/...). We compare
the JAX 2-m diagnostic to the Fortran 2-m diagnostic.

Coverage (the savepoint columns): vegetated (veg 5/9/10, FVEG~0.26-0.35) + bare/urban
(veg 13/16, FVEG=0 -> T2=T2MB) + daytime-unstable (cosz~0.94, soldn~800-1070) + the
STABLE-NOCTURNAL land columns (soldn=0; the +2.8 K crux the MYNN-SL empirical stand-in
was patching) + a Teide snow column.

WRF reports T2MV=0.0 for FVEG=0 columns (VEGE_FLUX is skipped, T2MV stays at its :2047
init); the JAX port always evaluates _vege_flux so its t2mv is a live (discarded) value
there. The combine ``where(use_veg, ...)`` takes ONLY t2mb when FVEG=0, exactly as the
WRF driver else-branch, so T2 is faithful regardless; this gate therefore compares t2mv
only where FVEG>0 and always compares t2mb + the combined t2.

Run (CPU only; cores 0-3):
    taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
        python3 proofs/v090/noahmp_t2mb_parity.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "proofs" / "noahmp"))

# Reuse the energy-gate column-feeding machinery (same WRF tables + column state).
import energy_savepoint_gate as eg  # noqa: E402

from gpuwrf.physics.noahmp.energy import noahmp_energy_canopy  # noqa: E402
from gpuwrf.physics.noahmp.energy_radiation import radiation_twostream  # noqa: E402

WRF_SOURCE = Path("/home/enric/src/wrf_pristine/WRF/phys/module_sf_noahmplsm.F")
SAVEPOINTS = ROOT / "proofs" / "noahmp" / "savepoints_energy.json"

# Predeclared fp64 tolerances [K]. The energy gate already shows TAH/TGB/SHG/SHC/SHB and
# the SFCDIF1 FV/CH match the oracle to <1e-4..1e-3 (fp64 transcription floor); T2MV/T2MB
# are direct algebraic functions of those plus FH2, so a 0.05 K absolute tol is a tight
# WRF-fidelity bar (NOT an obs/skill tol). T2 inherits the same.
TOL_K = 0.05


def run_column(col):
    """Run the JAX Noah-MP energy balance and return its 2-m LSM diagnostics."""
    ls = eg.build_state(col)
    forcing = eg.build_forcing(col)
    static = eg.build_static(col)
    phen = eg.build_phen(col)
    energy_p, rad_p = eg.build_params(col["vegtyp"], col["isltyp"])
    f = col["forcing"]
    co2 = 395.0e-6 * f["sfcprs"]
    o2 = 0.209 * f["sfcprs"]
    rad, extras = radiation_twostream(ls, forcing, static, phen, rad_p, eg.DT)
    _ls2, ef, _et = noahmp_energy_canopy(
        ls, forcing, static, rad, eg.DT, phen=phen, params=energy_p,
        rad_extras=extras, o2air=eg._f(o2), co2air=eg._f(co2), foln=eg._f(1.0),
        isurban=int(eg._P.isurban),
    )
    g = lambda a: float(np.asarray(a).reshape(-1)[0])  # noqa: E731
    return {"t2mv": g(ef.t2mv), "t2mb": g(ef.t2mb), "t2": g(ef.t2)}


def main():
    sp = json.load(open(SAVEPOINTS))
    cols = sp["columns"]
    src_sha = hashlib.sha256(WRF_SOURCE.read_bytes()).hexdigest()

    rows = []
    n_pass = n_fail = 0
    # residual accumulators for the stable-nocturnal subset (the crux)
    night_res = []
    for col in cols:
        wrf = col["wrf"]
        t2d = wrf.get("t2diag")
        if t2d is None:
            raise SystemExit("savepoints_energy.json missing t2diag — rebuild savepoints "
                             "(proofs/noahmp/build_noahmp_savepoints.py).")
        fveg = float(wrf["phen_out"]["fveg"])
        got = run_column(col)

        fields = {}
        # t2mv only where FVEG>0 (WRF reports 0.0 for bare; combine discards it).
        if fveg > 0.0:
            r = float(t2d["t2mv"]); v = got["t2mv"]
            fields["t2mv"] = (abs(v - r) <= TOL_K, r, v)
        for fld, key in (("t2mb", "t2mb"), ("t2", "t2m")):
            r = float(t2d[key]); v = got[fld]
            fields[fld] = (abs(v - r) <= TOL_K, r, v)

        col_ok = all(ok for ok, _, _ in fields.values())
        n_pass += col_ok
        n_fail += not col_ok
        is_night = col["case"] == "nighttime"
        if is_night:
            night_res.append((col["name"], fields["t2"][2] - fields["t2"][1]))
        rows.append((col["name"], col["case"], fveg, col_ok, fields))

    print(f"\n{'='*92}\nNoah-MP 2-m LSM DIAGNOSTIC (T2MV/T2MB/T2) REAL-WRF PARITY  "
          f"({len(cols)} columns, tol {TOL_K} K)\n{'='*92}")
    for name, case, fveg, col_ok, fields in rows:
        print(f"\n[{'PASS' if col_ok else 'FAIL'}] {name}  (case={case}, fveg={fveg:.3f})")
        for fld, (ok, r, v) in fields.items():
            mark = "ok " if ok else "XX "
            print(f"   {mark}{fld:5s} wrf={r:11.5f} K  jax={v:11.5f} K  d={v - r:+10.3e} K")
    print(f"\n{'-'*92}\nstable-nocturnal land T2 residuals (the +2.8 K crux the stand-in patched):")
    for name, dres in night_res:
        print(f"   {name:18s}  T2 jax-wrf = {dres:+.4f} K")
    worst = max((abs(v - r) for _, _, _, _, fl in rows for (_, r, v) in fl.values()), default=0.0)
    print(f"\n{'='*92}\nVERDICT: {n_pass} PASS / {n_fail} FAIL of {len(cols)} columns  "
          f"(worst |jax-wrf| = {worst:.3e} K)\n{'='*92}")

    proof = {
        "proof": "Noah-MP 2-m LSM diagnostic (T2MV/T2MB/combined T2) REAL-WRF parity (v0.9.0)",
        "kind": ("external oracle: compiled pristine WRF module_sf_noahmplsm.o NOAHMP_SFLX "
                 "T2MV/T2MB on real Canary d03 land columns vs JAX port; field-wise, "
                 "predeclared fp64 tol; NOT a self-compare"),
        "oracle": str(SAVEPOINTS.relative_to(ROOT)),
        "wrf_source": str(WRF_SOURCE),
        "wrf_source_sha256": src_sha,
        "wrf_refs": {
            "T2MV": "module_sf_noahmplsm.F:4148-4163 (VEGE_FLUX)",
            "T2MB": "module_sf_noahmplsm.F:4461-4474 (BARE_FLUX)",
            "combine": "module_sf_noahmplsm.F:2296/2311 + module_surface_driver.F:3469-3473",
        },
        "tol_K": TOL_K,
        "ncolumns": len(cols), "npass": n_pass, "nfail": n_fail,
        "worst_abs_residual_K": worst,
        "columns": [
            {"name": name, "case": case, "fveg": fveg, "pass": ok,
             "fields": {f: {"wrf_K": r, "jax_K": v, "abs_d_K": abs(v - r), "pass": p}
                        for f, (p, r, v) in fl.items()}}
            for name, case, fveg, ok, fl in rows
        ],
        "stable_nocturnal_T2_residual_K": {n: d for n, d in night_res},
        "verdict": "NOAHMP_T2MB_WRF_PARITY_PASS" if n_fail == 0 else "FAIL",
    }
    (HERE / "noahmp_t2mb_parity.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(f"proof -> {HERE / 'noahmp_t2mb_parity.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
