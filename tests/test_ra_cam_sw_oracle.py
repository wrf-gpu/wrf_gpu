"""CAM shortwave (ra_sw_physics = 3): radctl's SW branch vs pristine WRF V4.7.1.

Oracles (CPU, JAX_PLATFORMS=cpu):
* CAM01 (``proofs/cam_rad``): the pristine ``camrad``/``radctl`` true caller on real/augmented WN3 columns (committed
  compact fixture ``data/fixtures/cam01-compact-v1.npz``, 40 columns; full 210 via ``CAM01_FIXTURE``), with WRF's
  Registry aerosol indices P_SUL..P_VOLC = 2..13 (set_scalar_indices_from_config, package active).
* SWIDX (``proofs/cam_rad/swidx``, ``data/fixtures/cam-swidx-v1.npz``): the same pristine routines (aqsat,
  get_int_scales, get_aerosol, radinp, radcswmx) with the real-WRF indices on the CAM01 operands plus seeded synthetic
  columns that exercise the totwgt = 0 maximum-overlap pass, > 15 configurations with tied weights (findvalue) and very
  low sun.

Gate on e = max |port - WRF| / max(|WRF|, 1e-3 * max|WRF field|), by output class:
* fluxes, optical depths, rh, aerosol: e <= 1e-11 (measured <= 2.5e-13);
* differences of fluxes (diffuse = total - direct: fsdndif, fsdncdif, fsdsdif, solsd, solld; swcftoa = net - clear
  net): e <= 3e-11 (measured <= 1.1e-11);
* flux divergences (heating rates qrs, qrscs): e <= 1e-10 (measured <= 8.1e-11) AND the implied layer flux-divergence
  error <= 1e-12 of the column's incident flux (measured <= 1.9e-13).
The flux rounding difference itself is XLA:CPU's f64 exp (differs from glibc exp by <= 2 ulp in ~13 % of arguments;
sqrt identical; disabling XLA algsimp changes nothing); differences and divergences amplify it by 1/(relative size).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import gpuwrf  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp

from gpuwrf.physics import ra_cam_sw as sw
from gpuwrf.physics.ra_cam_common import CONST, radinp

_FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
CAM01 = Path(os.environ.get("CAM01_FIXTURE", _FIXTURES / "cam01-compact-v1.npz"))
SWIDX = Path(os.environ.get("CAM_SWIDX_FIXTURE", _FIXTURES / "cam-swidx-v1.npz"))

RTOL = 1e-11
RTOL_DIFF = 3e-11
RTOL_HEATING = 1e-10
DIV_TOL = 1e-12
DIFFERENCES = ("fsdndif", "fsdncdif", "fsdsdif", "solsd", "solld", "swcftoa")
FLOOR = 1e-3
OUTPUTS = ("qrs", "qrscs", "fsup", "fsupc", "fsdn", "fsdnc", "fsdndir", "fsdndif", "fsns", "fsds", "fsdsdir",
           "fsdsdif", "swcftoa", "sols", "soll", "solsd", "solld", "tauxcl", "tauxci")


def tol(name: str) -> float:
    if name in ("qrs", "qrscs"):
        return RTOL_HEATING
    return RTOL_DIFF if name in DIFFERENCES else RTOL


def rel_err(got, ref) -> float:
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    assert got.shape == ref.shape
    assert np.all(np.isfinite(got)), "non-finite port output"
    floor = FLOOR * max(float(np.max(np.abs(ref))), 1e-300)
    return float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), floor)))


def divergence_err(dq, pint_pa, coszrs, solcon) -> float:
    """Heating error as layer flux-divergence error (cgs) relative to the column incident flux (solin)."""

    dp = np.diff(np.asarray(pint_pa, np.float64) * 10.0, axis=1)
    solin = np.float32(solcon).astype(np.float64) * np.asarray(coszrs, np.float64) * 1000.0
    day = solin > 0
    ddiv = np.abs(np.asarray(dq, np.float64)) * dp / (1.e-4 * CONST.gravit)
    return float(np.max(ddiv[day] / solin[day, None]))


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"oracle fixture missing: {path}")
    return dict(np.load(path))


def _radctl_kwargs(d):
    return dict(q1=d["r8s_q1"], qliq=d["r8s_qliq"], qice=d["r8s_qice"], cld=d["r8s_cld"], pmid=d["r8s_pmid"],
                pint=d["r8s_pint"], t=d["r8s_t"], cicewp=d["r8s_cicewp"], cliqwp=d["r8s_cliqwp"], rel=d["r8s_rel"],
                rei=d["r8s_rei"], pmxrgn=d["r8s_pmxrgn"], nmxrgn=d["r8s_nmxrgn"], o3vmr=d["r8s_o3vmr"],
                coszrs=d["r8s_coszrs"], solcon=d["solcon"], albedo=d["in_albedo"], julian=d["in_julian"],
                mxaerl=np.int32(d["mxaerl"]), m_hybi=d["m_hybi"], co2mmr=d["r8s_co2mmr"], landfrac=None)


def _radctl(d, idx=sw.WRF_AEROSOL_INDICES):
    tables = sw.load_cam_aeropt_tables()
    fn = jax.jit(lambda kw, tb: sw.cam_sw_radctl(tables=tb, aer_idx=idx, **kw))
    out = fn(_radctl_kwargs(d), tables)
    return {k: np.asarray(v) for k, v in out.items()}


@pytest.fixture(scope="module")
def cam01():
    return _load(CAM01)


@pytest.fixture(scope="module")
def cam01_out(cam01):
    return _radctl(cam01)


def test_rh_and_aqsat(cam01):
    es, qs = sw.cam_aqsat(cam01["r8s_t"], cam01["r8s_pmid"])
    rh = sw.cam_rh(cam01["r8s_t"], cam01["r8s_pmid"], cam01["r8s_q1"])
    errs = {"esat": rel_err(es, cam01["swi_esat"]), "qsat": rel_err(qs, cam01["swi_qsat"]),
            "rh": rel_err(rh, cam01["swi_rh"])}
    print("aqsat/rh rel err", errs)
    assert max(errs.values()) <= RTOL, errs


def test_aerosol(cam01):
    aer = sw.cam_aerosol(cam01["r8s_pint"], cam01["in_julian"], int(cam01["mxaerl"]), cam01["m_hybi"],
                         idx=sw.WRF_AEROSOL_INDICES)
    ref = cam01["swi_aerosol"]
    err = rel_err(aer, ref)
    print("aerosol rel err", err, "nonzero slots", sorted({int(m) for m in np.nonzero(np.any(ref != 0, (0, 1)))[0]}))
    assert err <= RTOL
    assert np.array_equal(sw.get_int_scales(sw.WRF_AEROSOL_INDICES), cam01["swi_scales"][0])
    # all ten CAM aerosol species radiate (slots P_SUL..P_BCPHI = 2..11, 0-based 1..10)
    assert sorted({int(m) for m in np.nonzero(np.any(ref != 0, (0, 1)))[0]}) == list(range(1, 11))


def test_radctl_sw_outputs(cam01, cam01_out):
    errs = {k: rel_err(cam01_out[k], cam01["r8s_" + k]) for k in OUTPUTS}
    for k in OUTPUTS:
        print(f"{k:8s} max abs {np.max(np.abs(cam01_out[k] - cam01['r8s_' + k])):.3e} max rel {errs[k]:.3e}")
    divs = {k: divergence_err(cam01_out[k] - cam01["r8s_" + k], cam01["r8s_pint"], cam01["r8s_coszrs"],
                              cam01["solcon"]) for k in ("qrs", "qrscs")}
    print("heating as divergence error / incident flux", divs)
    bad = {k: v for k, v in errs.items() if not v <= tol(k)}
    assert not bad, bad
    assert max(divs.values()) <= DIV_TOL, divs
    night = cam01["r8s_coszrs"] <= 0.0
    assert night.any() and (~night).any()
    for k in OUTPUTS:
        assert np.all(cam01_out[k][night] == 0.0), k


def test_real_heating_rate(cam01, cam01_out):
    """camrad: RTHRATENSW = REAL(1.e4 * qrs / (cpair * pi_phy)) in WRF order."""

    pi8 = cam01["in_pi"].astype(np.float64)
    got = (1.e4 * cam01_out["qrs"][:, ::-1] / (CONST.cpair * pi8)).astype(np.float32)
    ref = cam01["s_rthratensw"]
    diff = got != ref
    ulps = np.abs(got.view(np.int32).astype(np.int64) - ref.view(np.int32).astype(np.int64))
    print(f"REAL heating: {int(diff.sum())} of {diff.size} float32 values differ, max {int(ulps.max())} ulp")
    assert int(ulps.max()) <= 1
    assert int(diff.sum()) <= max(2, diff.size // 10000)
    gsw = cam01_out["fsns"].astype(np.float32)
    assert int(np.sum(gsw != cam01["s_gsw"])) <= 1


@pytest.mark.parametrize("mutant", ["intended_indxsl", "no_aerosol", "argmin_findvalue"])
def test_deletion_sensitive_mutants(cam01, mutant, monkeypatch):
    """The gate must fail when the indxsl precision quirk is 'fixed', the aerosol is dropped, or WRF's findvalue
    quickselect (tie-breaking + persistent ptrc permutation) is replaced by a plain first-argmin."""

    if mutant == "intended_indxsl":
        intended = np.asarray([0] * 9 + [1] + [2] * 4 + [3] * 2 + [3] * 3, np.int32)   # what the comments intend
        assert not np.array_equal(intended, sw.INDXSL)
        monkeypatch.setattr(sw, "INDXSL", intended)
    elif mutant == "no_aerosol":
        real = sw.cam_aerosol
        monkeypatch.setattr(sw, "cam_aerosol", lambda *a, **k: 0.0 * real(*a, **k))
    else:
        monkeypatch.setattr(sw, "_findvalue1", lambda wgtv, ptrc, active: (jnp.argmin(wgtv).astype(jnp.int32), ptrc))
    out = _radctl(cam01)
    errs = {k: rel_err(out[k], cam01["r8s_" + k]) for k in ("qrs", "fsns", "fsdn", "sols")}
    print(mutant, errs)
    assert max(errs.values()) > 1e3 * RTOL


# ------------------------------------------------------------------------------------------------------------------- #
# SWIDX: real-WRF aerosol indices + synthetic overlap stress columns (radcswmx raw cgs outputs)                         #
# ------------------------------------------------------------------------------------------------------------------- #
_RAW = ("qrs", "qrscs", "fsup", "fsupc", "fsdn", "fsdnc", "fsdndir", "fsdndif", "fsdncdir", "fsdncdif", "tauxcl",
        "tauxci", "fsns", "fsntoa", "fsntoac", "fsds", "fsdsdir", "fsdsdif", "sols", "soll", "solsd", "solld")


def _swidx_run(d, idx):
    tables = sw.load_cam_aeropt_tables()

    def go(d, tb):
        pbr, pnm, _, o3mmr = radinp(d["in_pmid"], d["in_pint"], d["in_o3vmr"])
        rh = sw.cam_rh(d["in_t"], d["in_pmid"], d["in_q1"])
        aer = sw.cam_aerosol(d["in_pint"], d["in_julian"], d["mxaerl"], d["m_hybi"], idx=idx, tables=tb)
        alb = d["in_albedo"].astype(np.float64)
        out = sw.radcswmx(pnm=pnm, pbr=pbr, h2ommr=d["in_q1"], rh=rh, o3mmr=o3mmr, aermmr=aer, cld=d["in_cld"],
                          cicewp=d["in_cicewp"], cliqwp=d["in_cliqwp"], rel=d["in_rel"], rei=d["in_rei"],
                          coszrs=d["in_coszrs"], solcon=d["in_solcon"], asdir=alb, asdif=alb, aldir=alb, aldif=alb,
                          nmxrgn=d["in_nmxrgn"], pmxrgn=d["in_pmxrgn"], co2mmr=d["in_co2mmr"], tables=tb, idx=idx)
        return out, rh, aer

    keys = [k for k in d if k.startswith("in_")] + ["mxaerl", "m_hybi"]
    out, rh, aer = jax.jit(go)({k: d[k] for k in keys}, tables)
    return {k: np.asarray(v) for k, v in out.items()}, np.asarray(rh), np.asarray(aer)


def test_swidx_radcswmx():
    idx = sw.WRF_AEROSOL_INDICES
    d = _load(SWIDX)
    assert int(d["mode"]) == 1
    out, rh, aer = _swidx_run(d, idx)
    errs = {"rh": rel_err(rh, d["o_rh"]), "aerosol": rel_err(aer, d["o_aerosol"])}
    errs.update({k: rel_err(out[k], d["o_" + k]) for k in _RAW})
    for k, v in errs.items():
        print(f"{k:9s} max rel {v:.3e}")
    second = (d["o_nmx_after"] == 1) & (d["in_nmxrgn"] > 1)
    print("columns", len(d["columns"]), "day", int(np.sum(d["in_coszrs"] > 0)), "max-overlap second pass",
          int(second.sum()))
    assert second.sum() > 0, "no totwgt = 0 column exercised"
    divs = {k: divergence_err(out[k] - d["o_" + k], d["in_pint"], d["in_coszrs"], d["in_solcon"])
            for k in ("qrs", "qrscs")}
    print("heating as divergence error / incident flux", divs)
    bad = {k: v for k, v in errs.items() if not v <= tol(k)}
    assert not bad, bad
    assert max(divs.values()) <= DIV_TOL, divs
