"""B47: Noah-MP land Q2 (Q2V/Q2B + FVEG blend) vs the pristine WRF NOAHMP_SFLX column oracle.

Oracle: proofs/noahmp/savepoints_energy.json ``t2diag`` (q2v/q2b emitted by the compiled
pristine module_sf_noahmplsm.o driver, proofs/noahmp/noahmp_offline_driver.F90) on 11 real
Canary d03 land columns (vegetated day/night, bare/urban FVEG=0, Teide snow). The WRF land Q2
is formed as the surface driver does (module_surface_driver.F:3466/3471): Q2MV/Q2MB =
Q/(1-Q) (module_sf_noahmpdrv.F:1283-1284) blended by FVEG, FVEG=0 -> Q2MB.
Tolerance registered before measuring: |port - WRF| <= 1e-5 kg/kg (Q2 ~ 1e-2, ~0.1 %).
WRF leaves Q2V=0 where VEGE_FLUX is skipped (FVEG=0); q2v is compared only where FVEG>0.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
TOL = 1.0e-5


def _harness():
    path = ROOT / "proofs" / "v090" / "noahmp_t2mb_parity.py"
    spec = importlib.util.spec_from_file_location("noahmp_t2mb_parity_q2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _port_q2(h, col):
    from gpuwrf.physics.noahmp.energy import noahmp_energy_canopy
    from gpuwrf.physics.noahmp.energy_radiation import radiation_twostream
    eg = h.eg
    ls, forcing, static, phen = eg.build_state(col), eg.build_forcing(col), eg.build_static(col), eg.build_phen(col)
    energy_p, rad_p = eg.build_params(col["vegtyp"], col["isltyp"])
    f = col["forcing"]
    rad, extras = radiation_twostream(ls, forcing, static, phen, rad_p, eg.DT)
    _ls, ef, _et = noahmp_energy_canopy(
        ls, forcing, static, rad, eg.DT, phen=phen, params=energy_p, rad_extras=extras,
        o2air=eg._f(0.209 * f["sfcprs"]), co2air=eg._f(395.0e-6 * f["sfcprs"]), foln=eg._f(1.0),
        isurban=int(eg._P.isurban))
    g = lambda a: float(np.asarray(a).reshape(-1)[0])  # noqa: E731
    return {"q2v": g(ef.q2v), "q2b": g(ef.q2b), "q2": g(ef.q2), "fveg": g(phen.fveg)}


def _wrf_q2(t2d, fveg):
    mv, mb = t2d["q2v"] / (1.0 - t2d["q2v"]), t2d["q2b"] / (1.0 - t2d["q2b"])
    return fveg * mv + (1.0 - fveg) * mb if fveg > 0.0 else mb


FROZEN = Path(__file__).with_name("fixtures") / "savepoints_energy_frozen.json"


@pytest.mark.parametrize("native_real", ["0", "1"])
@pytest.mark.parametrize("corpus", ["warm11", "frozen4"])
def test_land_q2_matches_pristine_wrf_columns(monkeypatch, native_real, corpus):
    """warm11: frozen proofs fixture; frozen4: WN3 0227 d03 06Z cells with TG < TFRZ (HSUB
    branches of Q2B everywhere, of Q2V in the grassland column; review-b47 A2), built by the
    same pristine NOAHMP_SFLX offline driver."""
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", native_real)
    h = _harness()
    path = ROOT / "proofs" / "noahmp" / "savepoints_energy.json" if corpus == "warm11" else FROZEN
    cols = json.loads(path.read_text())["columns"]
    worst, bad = {}, []
    for col in cols:
        t2d = col["wrf"]["t2diag"]
        got = _port_q2(h, col)
        checks = {"q2b": (got["q2b"], t2d["q2b"]), "q2": (got["q2"], _wrf_q2(t2d, got["fveg"]))}
        if got["fveg"] > 0.0:
            checks["q2v"] = (got["q2v"], t2d["q2v"])
        for key, (port, wrf) in checks.items():
            err = abs(port - wrf)
            worst[key] = max(worst.get(key, 0.0), err)
            if not err <= TOL:
                bad.append((col["name"], key, port, wrf))
        if corpus == "frozen4":
            t2 = h.run_column(col)["t2"]
            if not abs(t2 - t2d["t2m"]) <= h.TOL_K:
                bad.append((col["name"], "t2", t2, t2d["t2m"]))
    assert len(cols) == {"warm11": 11, "frozen4": 4}[corpus] and not bad, (bad, worst)
