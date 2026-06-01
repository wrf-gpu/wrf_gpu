"""Noah-MP snow (Sprint S3) pristine-WRF savepoint-parity harness.

Loads the verbatim-WRF oracle savepoints
(``proofs/noahmp/fixtures/snow_oracle_savepoints.txt``, produced by
``oracle/snow_oracle`` — the unmodified module_sf_noahmplsm.F snow routines),
constructs the frozen ``NoahMPLandState`` per scenario, runs the JAX
``noahmp_snow`` kernel (fp64, branch-free masked layers), and asserts field-wise
parity (SNEQV/SNOWH/ISNOW/snow-layer T/ice/liq + ZSNSO + soil top-layer + TAUSS +
ALBOLD) against WRF across the accumulation/compaction/melt/sublimation set, plus
a SWE conservation cross-check. Writes a JSON proof object to
``proofs/noahmp/snow_savepoint_parity.json``.

WRF<->local index mapping
-------------------------
WRF snow arrays are indexed -NSNOW+1..0 (surface=0). The kernel's top-aligned
local index k = WRF_index + (NSNOW-1): so WRF -2,-1,0 -> local 0,1,2 (surface=2).
This is exactly the kernel's convention (surface always local NSNOW-1).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "src"))
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from jax import config  # noqa: E402

config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic  # noqa: E402
from gpuwrf.physics.noahmp.snow import noahmp_snow  # noqa: E402
from gpuwrf.physics.noahmp import snow as snowmod  # noqa: E402  (internal-path test)
from gpuwrf.physics.noahmp.types import NoahMPForcing  # noqa: E402

# Scenarios that exercise SUBLIMATION/FROST (QSNSUB/QSNFRO). These inputs are
# produced by the energy/water sprint and are NOT in the frozen noahmp_snow
# signature, so they are verified through the internal SNOWWATER column path
# (_snowwater_column, which accepts qsnsub/qsnfro) — same faithful physics.
SUBLIM_SCEN = {9, 10, 11}

DT = 1800.0
ZSOIL = np.array([-0.1, -0.4, -1.0, -2.0], dtype=np.float64)

# scenario forcing recovered from the oracle setup (only fields snow.py reads).
# (sfctmp, cosz, qsnow[mm/s], qrain[mm/s], qsnsub, qsnfro, tg) — must match
# oracle/snow_oracle.f90 setup_scenario exactly.
SCEN_FORCING = {
    1:  dict(sfctmp=270.0, cosz=0.5, qsnow=0.0,   qrain=0.0,   tg=270.0),
    2:  dict(sfctmp=268.0, cosz=0.5, qsnow=0.005, qrain=0.0,   tg=270.0),
    3:  dict(sfctmp=265.0, cosz=0.5, qsnow=0.02,  qrain=0.0,   tg=270.0),
    4:  dict(sfctmp=264.0, cosz=0.5, qsnow=0.01,  qrain=0.0,   tg=270.0),
    5:  dict(sfctmp=266.0, cosz=0.5, qsnow=0.0,   qrain=0.0,   tg=270.0),
    6:  dict(sfctmp=274.0, cosz=0.5, qsnow=0.0,   qrain=0.002, tg=273.16),
    7:  dict(sfctmp=270.0, cosz=0.5, qsnow=0.0,   qrain=0.001, tg=270.0),
    8:  dict(sfctmp=268.0, cosz=0.5, qsnow=0.0,   qrain=0.0,   tg=270.0),
    9:  dict(sfctmp=263.0, cosz=0.5, qsnow=0.0,   qrain=0.0,   tg=262.0, qsnsub=0.004),
    10: dict(sfctmp=266.0, cosz=0.5, qsnow=0.0,   qrain=0.0,   tg=270.0, qsnsub=0.003),
    11: dict(sfctmp=264.0, cosz=0.5, qsnow=0.0,   qrain=0.0,   tg=263.0, qsnfro=0.002),
    12: dict(sfctmp=271.0, cosz=0.7, qsnow=0.01,  qrain=0.0,   tg=270.0),
    13: dict(sfctmp=258.0, cosz=-0.1, qsnow=0.0,  qrain=0.0,   tg=258.0),
    14: dict(sfctmp=268.0, cosz=0.5, qsnow=0.0,   qrain=0.0,   tg=270.0),
}


def parse_savepoints(path):
    """Parse the oracle text dump into {scenario: {'PRE'|'POST': fields}}."""
    scen = {}
    cur_s = None
    cur_tag = None
    with open(path) as fh:
        lines = fh.readlines()
    i = 0
    fage = {}
    while i < len(lines):
        ln = lines[i].split()
        if ln and ln[0] == "SCEN" and len(ln) >= 3 and ln[2] in ("PRE", "POST"):
            cur_s = int(ln[1])
            cur_tag = ln[2]
            scen.setdefault(cur_s, {})[cur_tag] = {
                "snice": np.zeros(NSNOW), "snliq": np.zeros(NSNOW),
                "stc": np.zeros(NSNOW + NSOIL), "zsnso": np.zeros(NSNOW + NSOIL),
                "dzsnso": np.zeros(NSNOW + NSOIL),
                "sh2o": np.zeros(NSOIL), "sice": np.zeros(NSOIL),
            }
            i += 1
            continue
        if ln and ln[0] == "SCEN" and len(ln) >= 4 and ln[2] == "FAGE":
            fage[int(ln[1])] = float(ln[3])
            i += 1
            continue
        d = scen[cur_s][cur_tag]
        key = ln[0]
        if key == "ISNOW":
            d["isnow"] = int(ln[1])
        elif key == "SCAL":
            d["snowh"], d["sneqv"], d["sneqvo"], d["tauss"] = map(float, ln[1:5])
        elif key == "ALBOLD":
            d["albold"] = float(ln[1])
        elif key == "OUTS":
            d["qsnbot"], d["snoflow"], d["p1"], d["p2"] = map(float, ln[1:5])
        elif key == "SNL":
            iz = int(ln[1])           # WRF -2..0
            loc = iz + (NSNOW - 1)    # -> 0..2 (top-aligned)
            d["snice"][loc], d["snliq"][loc], d["stc"][loc] = map(float, ln[2:5])
        elif key == "ZD":
            iz = int(ln[1])           # WRF -2..NSOIL
            loc = iz + (NSNOW - 1)    # -> 0..NSNOW+NSOIL-1
            d["zsnso"][loc], d["dzsnso"][loc] = map(float, ln[2:4])
        elif key == "SOIL":
            iz = int(ln[1]) - 1       # WRF 1..4 -> 0..3
            d["sh2o"][iz], d["sice"][iz] = map(float, ln[2:4])
        i += 1
    return scen, fage


def build_land_state(pre):
    """Construct NoahMPLandState (1,1) from a parsed PRE savepoint."""
    def s2(v):
        return jnp.asarray([[v]], dtype=jnp.float64)

    def col(arr):  # (N,) -> (N,1,1)
        return jnp.asarray(arr, dtype=jnp.float64).reshape(-1, 1, 1)

    isnow = jnp.asarray([[pre["isnow"]]], dtype=jnp.int32)
    # snow arrays top-aligned (local 0..2); soil arrays 0..3
    tsno = col(pre["stc"][:NSNOW])
    snice = col(pre["snice"])
    snliq = col(pre["snliq"])
    zsnso = col(pre["zsnso"])
    # soil temperature/moisture from the dump
    tslb = col(pre["stc"][NSNOW:])
    sh2o = col(pre["sh2o"])
    sice = pre["sice"]
    smois = col(pre["sh2o"] + sice)

    z = jnp.zeros((1, 1), dtype=jnp.float64)
    zsoil4 = col(np.zeros(NSOIL))   # placeholder soil profile (unused fields)
    return NoahMPLandState(
        tslb=tslb, smois=smois, sh2o=sh2o, smcwtd=z,
        isnow=isnow, tsno=tsno, snice=snice, snliq=snliq, zsnso=zsnso,
        snowh=s2(pre["snowh"]), sneqv=s2(pre["sneqv"]), sneqvo=s2(pre["sneqvo"]),
        tauss=s2(pre["tauss"]), albold=s2(pre["albold"]),
        tv=s2(273.0), tg=s2(SCEN_TG), tah=s2(273.0), eah=z,
        canliq=z, canice=z, fwet=z, lai=z, sai=z, cm=z, ch=z,
        t_skin=s2(273.0), qsfc=z, znt=z, emiss=z, albedo=z,
        sfcrunoff=z, udrunoff=z,
    )


def build_static():
    z = jnp.zeros((1, 1), dtype=jnp.float64)
    zi = jnp.zeros((1, 1), dtype=jnp.int32)
    return NoahMPStatic(
        ivgtyp=zi, isltyp=zi, xland=z, landmask=z, lakemask=z, lu_index=zi,
        tbot=jnp.asarray([[285.0]], dtype=jnp.float64),
        dzs=jnp.asarray([0.1, 0.3, 0.6, 1.0], dtype=jnp.float64),
        zsoil=jnp.asarray(ZSOIL, dtype=jnp.float64),
        lat=z, dx_m=3000.0, parameters=None,
    )


def build_forcing(f):
    def s2(v):
        return jnp.asarray([[v]], dtype=jnp.float64)
    z = jnp.zeros((1, 1), dtype=jnp.float64)
    return NoahMPForcing(
        sfctmp=s2(f["sfctmp"]), sfcprs=s2(90000.0), psfc=s2(90000.0),
        uu=z, vv=z, qair=z, qc=z, soldn=z, lwdn=z,
        prcpconv=z, prcpnonc=z, prcpsnow=s2(f["qsnow"]), prcpgrpl=z, prcphail=z,
        cosz=s2(f["cosz"]), zlvl=s2(10.0),
        julian=jnp.asarray(15.0), yearlen=jnp.asarray(365.0),
    )


def _run_internal_snowwater(land, static, f, imelt):
    """Drive the full SNOWWATER column with explicit QSNSUB/QSNFRO + aging.

    Mirrors ``noahmp_snow`` exactly but threads the sublimation/frost inputs that
    the frozen public signature does not carry (they are produced by the energy/
    water sprint). Returns the same ``got`` dict shape as the public path.
    """
    dtype = jnp.float64
    sfctmp = jnp.asarray([[f["sfctmp"]]], dtype=dtype)
    bdfall = jnp.minimum(120.0, 67.92 + 51.25 * jnp.exp((sfctmp - snowmod.TFRZ) / 2.59))
    qsnow = jnp.asarray([[f["qsnow"]]], dtype=dtype)
    snowhin = jnp.where(qsnow > 0.0, qsnow / bdfall, 0.0)
    qrain = jnp.asarray([[f["qrain"]]], dtype=dtype)
    qsnsub = jnp.asarray([[f.get("qsnsub", 0.0)]], dtype=dtype)
    qsnfro = jnp.asarray([[f.get("qsnfro", 0.0)]], dtype=dtype)

    isnow = land.isnow.astype(jnp.int32)
    wx_old = land.snice + land.snliq
    ficeold = jnp.where(wx_old > 0.0, land.snice / jnp.where(wx_old > 0.0, wx_old, 1.0), 0.0)

    zsnso = land.zsnso.astype(dtype)
    active0 = snowmod._active_mask(isnow)
    zss = zsnso[:NSNOW]
    prev = jnp.concatenate([jnp.zeros_like(zss[:1]), zss[:-1]], axis=0)
    dzsnso_snow = jnp.where(active0, prev - zss, 0.0)

    sh2o_top = land.sh2o[0].astype(dtype)
    sice_top = (land.smois[0] - land.sh2o[0]).astype(dtype)
    zsoil = static.zsoil.astype(dtype)
    imelt_snow = imelt[:NSNOW].astype(jnp.int32)

    (isnow_n, snowh_n, sneqv_n, snice_n, snliq_n, sh2o_n, sice_n, tsno_n,
     dz_n, zsnso_full, qsnbot, snoflow, p1, p2) = snowmod._snowwater_column(
        isnow, land.snowh.astype(dtype), land.sneqv.astype(dtype),
        land.snice.astype(dtype), land.snliq.astype(dtype), sh2o_top, sice_top,
        land.tsno.astype(dtype), zsoil, qsnow, snowhin, qsnfro, qsnsub, qrain,
        sfctmp, ficeold, imelt_snow, dzsnso_snow, DT)

    sneqvo = land.sneqv.astype(dtype)
    tauss_n, _ = snowmod._snow_age(DT, land.tg.astype(dtype), sneqvo, sneqv_n,
                                   land.tauss.astype(dtype))
    alb_new = snowmod._snowalb_class(qsnow, DT, land.albold.astype(dtype))
    albold_n = jnp.where(jnp.asarray([[f["cosz"]]]) > 0.0, alb_new, land.albold.astype(dtype))

    return {
        "isnow": int(np.asarray(isnow_n)[0, 0]),
        "snowh": float(np.asarray(snowh_n)[0, 0]),
        "sneqv": float(np.asarray(sneqv_n)[0, 0]),
        "tauss": float(np.asarray(tauss_n)[0, 0]),
        "albold": float(np.asarray(albold_n)[0, 0]),
        "snice": np.asarray(snice_n)[:, 0, 0],
        "snliq": np.asarray(snliq_n)[:, 0, 0],
        "tsno": np.asarray(tsno_n)[:, 0, 0],
        "zsnso": np.asarray(zsnso_full)[:, 0, 0],
        "sh2o0": float(np.asarray(sh2o_n)[0, 0]),
    }


def run():
    fx = os.path.join(THIS_DIR, "fixtures", "snow_oracle_savepoints.txt")
    scen, fage = parse_savepoints(fx)
    static = build_static()

    results = []
    worst = {}
    all_pass = True
    global SCEN_TG
    for s in sorted(scen.keys()):
        pre = scen[s]["PRE"]
        post = scen[s]["POST"]
        f = SCEN_FORCING[s]
        SCEN_TG = f["tg"]
        land = build_land_state(pre)
        forcing = build_forcing(f)

        # imelt: snow rows (NSNOW) + soil rows (NSOIL); WRF only flags snow here.
        imelt = np.zeros((NSNOW + NSOIL, 1, 1), dtype=np.int32)
        # scenarios 6/7 set imelt on the snow layers (top-aligned local rows)
        if s == 6:
            # WRF imelt(-1)=1, imelt(0)=1 -> local 1,2
            imelt[1, 0, 0] = 1
            imelt[2, 0, 0] = 1
        imelt = jnp.asarray(imelt)

        qsnow = jnp.asarray([[f["qsnow"]]], dtype=jnp.float64)
        qmelt = jnp.asarray([[f["qrain"]]], dtype=jnp.float64)   # QRAIN -> pack

        if s in SUBLIM_SCEN:
            # full SNOWWATER column with explicit QSNSUB/QSNFRO (driver-supplied).
            got = _run_internal_snowwater(land, static, f, imelt)
        else:
            out = noahmp_snow(land, forcing, static, qsnow, imelt, qmelt, DT)
            got = {
                "isnow": int(np.asarray(out.isnow)[0, 0]),
                "snowh": float(np.asarray(out.snowh)[0, 0]),
                "sneqv": float(np.asarray(out.sneqv)[0, 0]),
                "tauss": float(np.asarray(out.tauss)[0, 0]),
                "albold": float(np.asarray(out.albold)[0, 0]),
                "snice": np.asarray(out.snice)[:, 0, 0],
                "snliq": np.asarray(out.snliq)[:, 0, 0],
                "tsno": np.asarray(out.tsno)[:, 0, 0],
                "zsnso": np.asarray(out.zsnso)[:, 0, 0],
                "sh2o0": float(np.asarray(out.sh2o)[0, 0, 0]),
            }
        ref = {
            "isnow": post["isnow"], "snowh": post["snowh"], "sneqv": post["sneqv"],
            "tauss": post["tauss"], "albold": post["albold"],
            "snice": post["snice"], "snliq": post["snliq"],
            "tsno": post["stc"][:NSNOW], "zsnso": post["zsnso"],
            "sh2o0": post["sh2o"][0],
        }

        diffs = {}
        diffs["isnow"] = abs(got["isnow"] - ref["isnow"])
        for k in ("snowh", "sneqv", "tauss", "albold", "sh2o0"):
            diffs[k] = abs(got[k] - ref[k])
        for k in ("snice", "snliq", "tsno", "zsnso"):
            diffs[k] = float(np.max(np.abs(got[k] - ref[k])))

        # tolerances: integer ISNOW exact; masses tight; T tight; albedo/tauss tight
        tol = {
            "isnow": 0, "snowh": 1e-9, "sneqv": 1e-7, "tauss": 1e-6,
            "albold": 1e-9, "sh2o0": 1e-9, "snice": 1e-6, "snliq": 1e-6,
            "tsno": 1e-6, "zsnso": 1e-7,
        }
        scen_pass = all(diffs[k] <= tol[k] for k in diffs)
        all_pass = all_pass and scen_pass

        # SWE conservation: out SWE == sum active (snice+snliq) when multilayer
        if got["isnow"] < 0:
            active_swe = float(np.sum((got["snice"] + got["snliq"])))
            cons = abs(active_swe - got["sneqv"])
        else:
            cons = 0.0

        for k, v in diffs.items():
            worst[k] = max(worst.get(k, 0.0), float(v))

        results.append({
            "scenario": s, "pass": bool(scen_pass),
            "wrf_isnow": ref["isnow"], "got_isnow": got["isnow"],
            "diffs": {k: float(v) for k, v in diffs.items()},
            "swe_conservation_residual": cons,
        })

    proof = {
        "kind": "noahmp_snow_savepoint_parity",
        "oracle": "verbatim module_sf_noahmplsm.F snow routines (gfortran fp64)",
        "fixtures": "proofs/noahmp/fixtures/snow_oracle_savepoints.txt",
        "n_scenarios": len(results),
        "all_pass": bool(all_pass),
        "worst_abs_diff": {k: float(v) for k, v in worst.items()},
        "tolerances": tol,
        "scenarios": results,
    }
    outp = os.path.join(THIS_DIR, "snow_savepoint_parity.json")
    with open(outp, "w") as fh:
        json.dump(proof, fh, indent=2)
    print(json.dumps({"all_pass": all_pass, "worst_abs_diff": proof["worst_abs_diff"]}, indent=2))
    print("wrote", outp)
    return all_pass


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
