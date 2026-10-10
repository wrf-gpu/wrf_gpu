"""Parity of the NSSL 2-moment diagnostics (radar reflectivity, effective radii) vs pristine WRF.

* radardd02: stage S5 -> S6 ``dbz2d`` (== DBZ_OUT) of the v034 column oracle, 14 cases
* effective radii (calc_eff_radius + driver clamps): S5 -> RE_CLOUD/ICE/SNOW_OUT
* adversarial point set from the dev harness ``cleanup_adv`` when its fixture is present.

fp64: <= 1e-12 relative (observed <= 4e-16: libm log10/pow rounding).  fp32: dBZ <= 4 REAL ulp
relative (observed <= 1.8e-7, log10f vs XLA log10), radii exact in REAL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import oracle_io as o  # noqa: E402

from gpuwrf.physics.nssl2mom import diagnostics  # noqa: E402
from gpuwrf.physics.nssl2mom.constants import get_constants  # noqa: E402
from gpuwrf.physics.nssl2mom.indices import FP32, FP64  # noqa: E402

PREC = {"fp64": FP64, "fp32": FP32}
RTOL_DBZ = {"fp64": 1e-12, "fp32": 4.8e-7}
RE_NAMES = ("RE_CLOUD_OUT", "RE_ICE_OUT", "RE_SNOW_OUT")
ADV_DIRS = (o.SAVEPOINTS, Path(os.environ.get("NSSL_O1_CLEANUP_ADV",
                                               "<USER_HOME>/wrf_gpu2_lanes/o1-nssl/cleanup_dev/adv")))


def _cast(x, mode):
    return np.asarray(x, PREC[mode].nR)


def _check(port, ref, rtol, label, atol=0.0):
    port = np.asarray(port, np.float64)
    ref = np.asarray(ref, np.float64)
    assert port.shape == ref.shape, label
    assert np.all(np.isfinite(port)), f"{label}: non-finite"
    err = np.abs(port - ref)
    bad = err > atol + rtol * np.abs(ref)
    if np.any(bad):
        i = int(np.argmax(np.where(bad, err, 0)))
        raise AssertionError(f"{label}: {int(bad.sum())} mismatches, worst {i}: port={port.flat[i]!r} ref={ref.flat[i]!r}")


def _dbz(sp, mode, C=None):
    return np.asarray(diagnostics.radardd02(_cast(o.stage_an(sp, "S5"), mode),
                                            _cast(o.stage_col(sp, "S0", "dn1"), mode),
                                            C or get_constants(mode), PREC[mode]))


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_radardd02_stage_parity(mode):
    nonzero = 0
    for case in o.CASES:
        sp = o.load_case(mode, case)
        ref = o.stage_col(sp, "S6", "dbz2d")
        # DBZ_OUT is the same field printed with 16 digits (ES23.15)
        _check(_cast(sp["columns"]["DBZ_OUT"], mode), _cast(ref, mode), 1e-15, f"{mode} case {case} DBZ_OUT")
        port = _dbz(sp, mode)
        _check(port, ref, RTOL_DBZ[mode], f"{mode} case {case} dbz")
        nonzero += int(np.sum(ref > 0))
    assert nonzero > 300  # rain, snow, ice, graupel and hail reflectivity all present


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_eff_radius_parity(mode):
    for case in o.CASES:
        sp = o.load_case(mode, case)
        res = diagnostics.eff_radius(_cast(o.stage_an(sp, "S5"), mode),
                                     _cast(o.stage_col(sp, "S0", "dn1"), mode),
                                     get_constants(mode), PREC[mode])
        for name, port in zip(RE_NAMES, res):
            ref = _cast(sp["columns"][name], mode)  # ES23.15 print: exact for REAL, 1e-16 for DOUBLE
            _check(np.asarray(port), ref, 1e-14 if mode == "fp64" else 0.0, f"{mode} case {case} {name}")


def test_radardd02_mutant_snow_coefficient_detected():
    """Deletion sensitivity (E39): a wrong snow reflectivity constant must break parity."""
    mode = "fp64"
    C = get_constants(mode)
    mut = SimpleNamespace(**vars(C))
    mut.gsnow73 = C.gsnow73 * 1.01
    with pytest.raises(AssertionError):
        for case in o.CASES:
            sp = o.load_case(mode, case)
            _check(_dbz(sp, mode, mut), o.stage_col(sp, "S6", "dbz2d"), RTOL_DBZ[mode], "mutant")


# ------------------------------------------------------------------ adversarial point set
def _adv(mode):
    for d in ADV_DIRS:
        p = Path(d) / f"cleanup_adversarial_{mode}.json"
        if p.exists():
            return json.loads(p.read_text())
    pytest.skip(f"adversarial fixture cleanup_adversarial_{mode}.json not found in {ADV_DIRS}")


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_radardd02_adversarial(mode):
    fx = _adv(mode)
    an = np.zeros((19, fx["n"]))
    an[1:] = np.asarray(fx["an_in"])
    port = diagnostics.radardd02(_cast(an, mode), _cast(fx["dn"], mode), get_constants(mode), PREC[mode])
    _check(port, np.asarray(fx["dbz"]), RTOL_DBZ[mode], f"{mode} adversarial dbz")


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_eff_radius_adversarial(mode):
    fx = _adv(mode)
    an = np.zeros((19, fx["n"]))
    an[1:] = np.asarray(fx["an_in"])
    res = diagnostics.eff_radius(_cast(an, mode), _cast(fx["dn"], mode), get_constants(mode), PREC[mode])
    for j, port in enumerate(res):
        _check(port, np.asarray(fx["re"])[:, j], 1e-14 if mode == "fp64" else 0.0,
               f"{mode} adversarial {RE_NAMES[j]}")
