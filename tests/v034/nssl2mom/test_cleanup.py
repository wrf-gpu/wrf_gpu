"""Parity of the NSSL 2-moment cleanup routines against the pristine-WRF oracle.

* calcnfromq: stage S0 -> S1 of the v034 column oracle (cold-start cases, itimestep == 1)
* smallvalues: stage S4 -> S5 (all 14 cases)
* adversarial point set (every reachable branch of both routines) from the pristine module
  driven by a dev harness (``cleanup_adv``), when its fixture is present.

fp64 port vs ``-fdefault-real-8`` oracle: |port - ref| <= 1e-10 |ref| (observed: bitwise).
fp32 port vs WRF REAL oracle: <= 1 REAL ulp relative band (observed: bitwise).
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

from gpuwrf.physics.nssl2mom import cleanup  # noqa: E402
from gpuwrf.physics.nssl2mom.constants import get_constants  # noqa: E402
from gpuwrf.physics.nssl2mom.indices import FP32, FP64, LCCNA, LHL, LT  # noqa: E402

PREC = {"fp64": FP64, "fp32": FP32}
RTOL = {"fp64": 1e-10, "fp32": 1.2e-7}
ADV_NAMES = ("cleanup_adversarial_{mode}.json",)
ADV_DIRS = (o.SAVEPOINTS, Path(os.environ.get("NSSL_O1_CLEANUP_ADV",
                                               "<USER_HOME>/wrf_gpu2_lanes/o1-nssl/cleanup_dev/adv")))


def _check(port, ref, mode, label, atol=0.0):
    port = np.asarray(port, np.float64)
    ref = np.asarray(ref, np.float64)
    assert port.shape == ref.shape, label
    assert np.all(np.isfinite(port)), f"{label}: non-finite port values"
    err = np.abs(port - ref)
    bad = err > atol + RTOL[mode] * np.abs(ref)
    if np.any(bad):
        i = np.unravel_index(np.argmax(np.where(bad, err / np.maximum(np.abs(ref), 1e-300), 0)), err.shape)
        raise AssertionError(f"{label}: {int(bad.sum())} mismatches, worst at {i}: port={port[i]!r} ref={ref[i]!r}")


def _cast(x, mode):
    return np.asarray(x, PREC[mode].nR)


def _run_calcnfromq(sp, mode, C=None):
    C = C or get_constants(mode)
    a0 = o.stage_an(sp, "S0")
    dn = o.stage_col(sp, "S0", "dn1")
    return np.asarray(cleanup.calcnfromq(_cast(a0, mode), _cast(dn, mode), C, PREC[mode]))


def _run_smallvalues(sp, mode, C=None):
    C = C or get_constants(mode)
    a4 = o.stage_an(sp, "S4")
    an, t0 = cleanup.smallvalues(_cast(a4, mode), _cast(o.stage_col(sp, "S4", "t0"), mode),
                                 _cast(o.stage_col(sp, "S0", "dn1"), mode), None,
                                 _cast(o.stage_col(sp, "S0", "t77"), mode), sp["scalars"]["DT"],
                                 C, PREC[mode])
    return np.asarray(an), np.asarray(t0)


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_calcnfromq_stage_parity(mode):
    ncold = 0
    for case in o.CASES:
        sp = o.load_case(mode, case)
        if sp["scalars"]["ITIMESTEP"] != 1:
            continue
        ncold += 1
        out = _run_calcnfromq(sp, mode)
        _check(out[1:], o.stage_an(sp, "S1")[1:], mode, f"{mode} case {case} calcnfromq")
    assert ncold == 7


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_warm_start_skips_calcnfromq(mode):
    """The driver calls calcnfromq only at itimestep == 1: warm-start oracle S1 == S0."""
    for case in o.CASES:
        sp = o.load_case(mode, case)
        if sp["scalars"]["ITIMESTEP"] == 1:
            continue
        np.testing.assert_array_equal(o.stage_an(sp, "S1"), o.stage_an(sp, "S0"))


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_smallvalues_stage_parity(mode):
    for case in o.CASES:
        sp = o.load_case(mode, case)
        an, t0 = _run_smallvalues(sp, mode)
        _check(an[1:], o.stage_an(sp, "S5")[1:], mode, f"{mode} case {case} smallvalues")
        # t0 is reset from theta*pii before the cleanup (module line 12110)
        _check(t0, an[LT] * o.stage_col(sp, "S0", "t77"), mode, f"{mode} case {case} t0", atol=0.0)


def test_smallvalues_mutant_ccn_restore_detected():
    """Deletion sensitivity (E39): a wrong CCN restore time constant must break stage parity."""
    mode = "fp64"
    C = get_constants(mode)
    mut = SimpleNamespace(**vars(C))
    mut.ccntimeconst = C.ccntimeconst * 1.5
    hit = 0
    for case in o.CASES:
        sp = o.load_case(mode, case)
        an, _ = _run_smallvalues(sp, mode, mut)
        if np.any(an[LCCNA] != o.stage_an(sp, "S5")[LCCNA]):
            hit += 1
    assert hit >= 2


def test_calcnfromq_mutant_fold_threshold_detected():
    mode = "fp64"
    C = get_constants(mode)
    mut = SimpleNamespace(**vars(C))
    mut.qxmin = C.qxmin.copy()
    mut.qxmin[LHL] = 0.0
    mut.qxmin_init = C.qxmin_init.copy()
    mut.qxmin_init[LHL] = 0.0
    out = _run_calcnfromq(o.load_case(mode, 7), mode, mut)
    assert np.any(out[LHL] != o.stage_an(o.load_case(mode, 7), "S1")[LHL])


# ------------------------------------------------------------------ adversarial point set
def _adv(mode):
    for d in ADV_DIRS:
        for n in ADV_NAMES:
            p = Path(d) / n.format(mode=mode)
            if p.exists():
                return json.loads(p.read_text())
    pytest.skip(f"adversarial cleanup fixture cleanup_adversarial_{mode}.json not found in {ADV_DIRS} "
                "(build with cleanup_dev/adv/build_run.sh + make_fixture.py)")


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_calcnfromq_adversarial(mode):
    fx = _adv(mode)
    an = np.zeros((19, fx["n"]))
    an[1:] = np.asarray(fx["an_in"])
    out = np.asarray(cleanup.calcnfromq(_cast(an, mode), _cast(fx["dn"], mode), get_constants(mode), PREC[mode]))
    _check(out[1:], np.asarray(fx["calcnfromq"]), mode, f"{mode} adversarial calcnfromq")


@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_smallvalues_adversarial(mode):
    fx = _adv(mode)
    an = np.zeros((19, fx["n"]))
    an[1:] = np.asarray(fx["an_in"])
    t77 = _cast(fx["t77"], mode)
    out, t0 = cleanup.smallvalues(_cast(an, mode), None, _cast(fx["dn"], mode), None, t77,
                                  fx["dtp"], get_constants(mode), PREC[mode])
    _check(np.asarray(out)[1:], np.asarray(fx["smallvalues"]), mode, f"{mode} adversarial smallvalues")
    _check(t0, np.asarray(fx["smallvalues_t0"]), mode, f"{mode} adversarial smallvalues t0")
