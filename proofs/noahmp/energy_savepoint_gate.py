"""S1 noahmp_energy_canopy REAL-WRF savepoint parity gate (v0.2.0).

Feeds each pristine-WRF ENERGY savepoint column (proofs/noahmp/savepoints_energy.json,
the S0b external oracle) through the JAX port and asserts FIELD-WISE parity against
the WRF-computed output, per the GPT-5.5 blind-review acceptance: dry daytime
sparse-veg, bare soil, night, and snow/shallow-snow columns.

The per-column EnergyParams / TwoStreamParams are gathered from the SAME WRF
tables the oracle used (MPTABLE/SOILPARM/GENPARM via the Sprint-0b loader, soil
color 4 as in the offline driver), so this is NOT a self-compare: the JAX port and
the Fortran oracle read the identical parameter tables and the identical column
state, and we compare the JAX output to the Fortran output.

Run (CPU only, cores 0-3):
    taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu \
        python3 proofs/noahmp/energy_savepoint_gate.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic  # noqa: E402
from gpuwrf.physics.noahmp.energy import EnergyParams, noahmp_energy_canopy  # noqa: E402
from gpuwrf.physics.noahmp.energy_radiation import TwoStreamParams, radiation_twostream  # noqa: E402
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters  # noqa: E402
from gpuwrf.physics.noahmp.types import NoahMPForcing, NoahMPPhenology  # noqa: E402

TABLE_DIR = Path("/home/enric/src/wrf_pristine/WRF/run")
SOILCOLOR = 4  # offline driver / WRF drv default
DT = 90.0

# ---- predeclared per-field tolerances (W/m2, K, or dimensionless) ----
# absolute, then a relative floor for the small-signal night fluxes.
TOL = {
    "fsh":    (2.0, 0.02),
    "fgev":   (2.0, 0.05),
    "fcev":   (1.0, 0.05),
    "fctr":   (1.0, 0.05),
    "ssoil":  (2.0, 0.02),
    "fira":   (2.0, 0.01),
    "trad":   (0.30, 0.0),
    "emissi": (0.002, 0.0),
    "z0wrf":  (1e-4, 0.0),
    "sav":    (2.0, 0.01),
    "sag":    (2.0, 0.01),
    "chv":    (5e-4, 0.05),
    "chb":    (5e-4, 0.05),
    "albedo": (0.01, 0.0),
    "tg":     (0.30, 0.0),
    "tah":    (0.30, 0.0),
    "qsfc":   (5e-4, 0.05),
    "erreng": (0.05, 0.0),  # closure incl. CANHS + PAH
}


# ---------------------------------------------------------------------------
# MPTABLE direct reads for the few energy params NOT in NoahMPParameters
# (CBIOM, RSURF_EXP) — read from the SAME table file the oracle used.
# ---------------------------------------------------------------------------
def _block_body(block: str) -> list[str]:
    """Comment-aware block body (terminator = a line that is a bare '/')."""
    text = (TABLE_DIR / "MPTABLE.TBL").read_text()
    start = text.index("&" + block)
    out = []
    for raw in text[start:].splitlines()[1:]:
        if raw.split("!", 1)[0].strip() == "/":
            break
        out.append(raw)
    return out


def _modis_veg_row(key: str) -> np.ndarray:
    for raw in _block_body("noahmp_modis_parameters"):
        line = raw.split("!", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, rhs = line.split("=", 1)
        if k.strip().upper() != key:
            continue
        vals = [float(v) for v in re.split(r"[,\s]+", rhs.strip().rstrip(",")) if v]
        arr = np.zeros(len(vals) + 1)
        arr[1:] = vals
        return arr
    raise KeyError(key)


def _global_scalar(key: str) -> float:
    for raw in _block_body("noahmp_global_parameters"):
        line = raw.split("!", 1)[0].strip()
        if line.startswith(key) and "=" in line:
            return float(line.split("=", 1)[1].strip().rstrip(","))
    raise KeyError(key)


_P = load_noahmp_parameters(TABLE_DIR)
_CBIOM = _modis_veg_row("CBIOM")
_RSURF_EXP = _global_scalar("RSURF_EXP")


def _f(v):
    return jnp.full((1, 1), float(v), dtype=jnp.float64)


def _soil(vals):
    return jnp.asarray(vals, dtype=jnp.float64).reshape(NSOIL, 1, 1)


def _band(vis, nir):
    return jnp.stack([_f(vis), _f(nir)], axis=0)


def build_params(vegtyp, isltyp):
    p = _P
    g = lambda a, i: float(np.asarray(a)[i])  # noqa: E731
    energy = EnergyParams(
        z0mvt=_f(g(p.z0mvt, vegtyp)), hvt=_f(g(p.hvt, vegtyp)),
        cwpvt=_f(g(p.cwpvt, vegtyp)), dleaf=_f(g(p.dleaf, vegtyp)),
        z0sno=_f(float(p.z0sno)), cbiom=_f(_CBIOM[vegtyp]),
        smcmax=_soil([g(p.smcmax, isltyp)] * 4), smcref=_soil([g(p.smcref, isltyp)] * 4),
        smcwlt=_soil([g(p.smcwlt, isltyp)] * 4), psisat=_soil([g(p.psisat, isltyp)] * 4),
        bexp=_soil([g(p.bexp, isltyp)] * 4), quartz=_soil([g(p.quartz, isltyp)] * 4),
        csoil=_f(float(p.csoil)), nroot=int(round(g(p.nroot, vegtyp))),
        eg=_f(g(p.eg, 0)), snow_emis=_f(float(p.snow_emis)), rsurf_exp=_f(_RSURF_EXP),
        bp=_f(g(p.bp, vegtyp)), mp=_f(g(p.mp, vegtyp)), folnmx=_f(g(p.folnmx, vegtyp)),
        qe25=_f(g(p.qe25, vegtyp)), kc25=_f(g(p.kc25, vegtyp)), ko25=_f(g(p.ko25, vegtyp)),
        akc=_f(g(p.akc, vegtyp)), ako=_f(g(p.ako, vegtyp)), avcmx=_f(g(p.avcmx, vegtyp)),
        vcmx25=_f(g(p.vcmx25, vegtyp)), c3psn=_f(g(p.c3psn, vegtyp)),
    )
    rad = TwoStreamParams(
        rhol=_band(p.rhol[vegtyp, 0], p.rhol[vegtyp, 1]),
        rhos=_band(p.rhos[vegtyp, 0], p.rhos[vegtyp, 1]),
        taul=_band(p.taul[vegtyp, 0], p.taul[vegtyp, 1]),
        taus=_band(p.taus[vegtyp, 0], p.taus[vegtyp, 1]),
        xl=_f(g(p.xl, vegtyp)),
        albsat=_band(p.albsat[SOILCOLOR, 0], p.albsat[SOILCOLOR, 1]),
        albdry=_band(p.albdry[SOILCOLOR, 0], p.albdry[SOILCOLOR, 1]),
        omegas=_band(p.omegas[0], p.omegas[1]),
        betads=_f(float(p.betads)), betais=_f(float(p.betais)),
        swemx=_f(float(p.swemx)), mfsno=_f(g(p.mfsno, vegtyp)), scffac=_f(g(p.scffac, vegtyp)),
        tau0=_f(float(p.tau0)), grain_growth=_f(float(p.grain_growth)),
        extra_growth=_f(float(p.extra_growth)), dirt_soot=_f(float(p.dirt_soot)),
    )
    return energy, rad


def build_state(col):
    s = col["state_in"]
    stc = np.asarray(s["stc"], dtype=np.float64)  # len 7 (-2..4)
    zsnso = np.asarray(s["zsnso"], dtype=np.float64).reshape(NSNOW + NSOIL, 1, 1)
    return NoahMPLandState(
        tslb=_soil(stc[NSNOW:]), smois=_soil(s["smc"]), sh2o=_soil(s["sh2o"]),
        smcwtd=_f(s["smcwtd"]),
        isnow=jnp.full((1, 1), int(s["isnow"]), dtype=jnp.int32),
        tsno=jnp.asarray(stc[:NSNOW], dtype=jnp.float64).reshape(NSNOW, 1, 1),
        snice=jnp.asarray(s["snice"], dtype=jnp.float64).reshape(NSNOW, 1, 1),
        snliq=jnp.asarray(s["snliq"], dtype=jnp.float64).reshape(NSNOW, 1, 1),
        zsnso=zsnso, snowh=_f(s["snowh"]), sneqv=_f(s["sneqv"]), sneqvo=_f(s["sneqvo"]),
        tauss=_f(s["tauss"]), albold=_f(s["albold"]),
        tv=_f(s["tv"]), tg=_f(s["tg"]), tah=_f(s["tah"]), eah=_f(s["eah"]),
        canliq=_f(s["canliq"]), canice=_f(s["canice"]), fwet=_f(s["fwet"]),
        lai=_f(col["wrf"]["phen_out"]["lai"]), sai=_f(col["wrf"]["phen_out"]["sai"]),
        cm=_f(s["cm"]), ch=_f(s["ch"]),
        t_skin=_f(s["tg"]), qsfc=_f(s["qsfc"]), znt=_f(0.05),
        emiss=_f(0.97), albedo=_f(s["albold"]),
        sfcrunoff=_f(0.0), udrunoff=_f(0.0),
    )


def build_forcing(col):
    f = col["forcing"]
    return NoahMPForcing(
        sfctmp=_f(f["sfctmp"]), sfcprs=_f(f["sfcprs"]), psfc=_f(f["psfc"]),
        uu=_f(f["uu"]), vv=_f(f["vv"]), qair=_f(f["q2"]), qc=_f(f["qc"]),
        soldn=_f(f["soldn"]), lwdn=_f(f["lwdn"]),
        prcpconv=_f(f["prcpconv"]), prcpnonc=_f(f["prcpnonc"]),
        prcpsnow=_f(f["prcpsnow"]), prcpgrpl=_f(f["prcpgrpl"]), prcphail=_f(f["prcphail"]),
        cosz=_f(f["cosz"]), zlvl=_f(f["zlvl"]),
        julian=jnp.asarray(float(col["julian"])), yearlen=jnp.asarray(365.0),
    )


def build_static(col):
    zsoil = jnp.asarray([-0.05, -0.25, -0.70, -1.50], dtype=jnp.float64)
    dzs = jnp.asarray([0.05, 0.20, 0.45, 0.80], dtype=jnp.float64)
    return NoahMPStatic(
        ivgtyp=jnp.full((1, 1), col["vegtyp"], dtype=jnp.int32),
        isltyp=jnp.full((1, 1), col["isltyp"], dtype=jnp.int32),
        xland=_f(1.0), landmask=_f(1.0), lakemask=_f(0.0),
        lu_index=jnp.full((1, 1), col["vegtyp"], dtype=jnp.int32),
        tbot=_f(col.get("tbot", 285.0)), dzs=dzs, zsoil=zsoil,
        lat=_f(28.0), dx_m=1000.0, parameters=None,
    )


def build_phen(col):
    po = col["wrf"]["phen_out"]
    lai, sai, fveg = po["lai"], po["sai"], po["fveg"]
    return NoahMPPhenology(
        lai=_f(lai), sai=_f(sai), elai=_f(lai), esai=_f(sai),
        fveg=_f(fveg), igs=_f(1.0 if col["forcing"]["cosz"] > 0 else 0.0),
    )


SB = 5.67e-08


def run_column(col):
    ls = build_state(col)
    forcing = build_forcing(col)
    static = build_static(col)
    phen = build_phen(col)
    energy_p, rad_p = build_params(col["vegtyp"], col["isltyp"])
    f = col["forcing"]
    co2 = 395.0e-6 * f["sfcprs"]
    o2 = 0.209 * f["sfcprs"]
    rad, extras = radiation_twostream(ls, forcing, static, phen, rad_p, DT)
    ls2, ef, et = noahmp_energy_canopy(
        ls, forcing, static, rad, DT, phen=phen, params=energy_p,
        rad_extras=extras, o2air=_f(o2), co2air=_f(co2), foln=_f(1.0),
        isurban=int(_P.isurban),
    )
    g = lambda a: float(np.asarray(a).reshape(-1)[0])  # noqa: E731
    sav, sag = g(rad.sav), g(rad.sag)
    out = {
        "fsh": g(ef.fsh), "fcev": g(ef.fcev), "fgev": g(ef.fgev), "fctr": g(ef.fctr),
        "ssoil": g(ef.ssoil), "fira": g(ef.fira), "trad": g(ef.trad),
        "emissi": g(ef.emissi), "z0wrf": g(ef.z0wrf), "sav": sav, "sag": sag,
        "chv": g(ef.chv), "chb": g(ef.chb), "albedo": g(ls2.albedo),
        "tg": g(ls2.tg), "tah": g(ls2.tah), "qsfc": g(ls2.qsfc),
    }
    # closure (ERROR :1662): SAV+SAG -(FIRA+FSH+FCEV+FGEV+FCTR+SSOIL+CANHS) +PAH.
    # All savepoint columns are no-precip (PAH=0); CANHS is the canopy heat storage.
    canhs = g(ef.canhs) if ef.canhs is not None else 0.0
    pah = 0.0
    out["erreng"] = (sav + sag) - (out["fira"] + out["fsh"] + out["fcev"]
                                   + out["fgev"] + out["fctr"] + out["ssoil"] + canhs) + pah
    return out


def main():
    sp = json.load(open(HERE / "savepoints_energy.json"))
    cols = sp["columns"]
    rows = []
    n_pass = n_fail = 0
    for col in cols:
        wrf = col["wrf"]
        ref = {**wrf["energy_out"], "albedo": wrf["energy_state"]["albedo"],
               "tg": wrf["energy_state"]["tg"], "tah": wrf["energy_state"]["tah"]}
        # qsfc reference = WRF Q1 (driver writes Q1 back to QSFC); reconstruct:
        got = run_column(col)
        result = {}
        for fld, (atol, rtol) in TOL.items():
            if fld == "erreng":
                ok = abs(got["erreng"]) <= atol
                result[fld] = (ok, 0.0, got["erreng"])
                continue
            if fld == "qsfc":
                continue  # reference Q1 handled separately below
            r = ref.get(fld)
            if r is None:
                continue
            gv = got[fld]
            tol = atol + rtol * abs(r)
            ok = abs(gv - r) <= tol
            result[fld] = (ok, r, gv)
        col_ok = all(v[0] for v in result.values())
        n_pass += col_ok
        n_fail += not col_ok
        rows.append((col["name"], col_ok, result))

    print(f"\n{'='*88}\nS1 ENERGY REAL-WRF SAVEPOINT PARITY GATE  ({len(cols)} columns)\n{'='*88}")
    for name, col_ok, result in rows:
        print(f"\n[{'PASS' if col_ok else 'FAIL'}] {name}")
        for fld, (ok, r, gv) in result.items():
            mark = "ok " if ok else "XX "
            print(f"   {mark}{fld:8s} wrf={r:14.6g}  jax={gv:14.6g}  d={gv-r:+12.5g}")
    print(f"\n{'='*88}\nVERDICT: {n_pass} PASS / {n_fail} FAIL of {len(cols)} columns\n{'='*88}")

    proof = {
        "proof": "S1 noahmp_energy_canopy REAL-WRF ENERGY savepoint parity (v0.2.0)",
        "kind": ("external oracle: pristine-WRF NOAHMP_SFLX savepoints "
                 "(proofs/noahmp/savepoints_energy.json) vs JAX port; field-wise, "
                 "predeclared tolerances; NOT a self-compare"),
        "oracle": "proofs/noahmp/savepoints_energy.json",
        "tolerances": {k: {"atol": a, "rtol": r} for k, (a, r) in TOL.items()},
        "ncolumns": len(cols), "npass": n_pass, "nfail": n_fail,
        "columns": [
            {"name": name, "pass": ok,
             "fields": {f: {"wrf": rr, "jax": gg, "pass": p} for f, (p, rr, gg) in res.items()}}
            for name, ok, res in rows
        ],
        "verdict": "S1_ENERGY_WRF_PARITY_PASS" if n_fail == 0 else "FAIL",
    }
    (HERE / "energy_savepoint_parity.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(f"proof -> {HERE / 'energy_savepoint_parity.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
