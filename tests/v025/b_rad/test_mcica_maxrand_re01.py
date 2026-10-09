"""RE01: GPUWRF_RRTMG_MAXRAND on fid-q2's pristine true-caller fixture (186 real WN3 0227 columns).

Flag ON with the port's constant radii must be pristine arm C (cldovrlp = 2, 10/30/75 um) at the frozen tier-1 bounds
(scalars + surface..model-top interfaces + heating); flag OFF stays arm D (fid-q2's port-today test); clear columns are
bitwise invariant under the flag; deleting the chain (flag on, identity overlap) falls back to D and misses C.
CPU legacy column solvers, fid-q2's own runner (tests/v025/fid_q2/test_rrtmg_mp_re_port_today.py).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.kernels import rad_mcica
from gpuwrf.physics import rrtmg_lw, rrtmg_sw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fid_q2"))
from rrtmg_mp_re_fixture import FIXTURE, columns_outside, load  # noqa: E402
from test_rrtmg_mp_re_port_today import _port_today  # noqa: E402


def _run(monkeypatch, *, maxrand, chain=None):
    monkeypatch.setattr(rad_mcica, "_MAXRAND", maxrand)
    if chain is not None:
        monkeypatch.setattr(rad_mcica, "max_random_cdf", chain)
    for fn in (rrtmg_lw.solve_rrtmg_lw_column, rrtmg_sw.solve_rrtmg_sw_column):
        fn.clear_cache()  # the flag is read at trace time
    try:
        return _port_today(load(FIXTURE))
    finally:
        for fn in (rrtmg_lw.solve_rrtmg_lw_column, rrtmg_sw.solve_rrtmg_sw_column):
            fn.clear_cache()


@pytest.fixture(scope="module")
def runs():
    fx = load(FIXTURE)
    mp = pytest.MonkeyPatch()
    try:
        on = _run(mp, maxrand=True)
        deleted = _run(mp, maxrand=True, chain=lambda random, cldf: random)
        mp.undo()
        off = _run(mp, maxrand=False)
    finally:
        mp.undo()
    return fx, on, off, deleted


def test_maxrand_is_pristine_arm_c_at_frozen_bounds(runs):
    fx, on, _, _ = runs
    outside = columns_outside(fx, on, reference_arm="C")
    assert not outside.any(), np.flatnonzero(outside)


def test_maxrand_off_stays_arm_d_and_clear_columns_are_bitwise(runs):
    fx, on, off, _ = runs
    assert not columns_outside(fx, off, reference_arm="D").any()
    clear = fx["category"] == "clear"
    for key in on:
        np.testing.assert_array_equal(on[key][clear], off[key][clear], err_msg=key)
    assert (on["glw"] != off["glw"])[~clear].sum() >= 100  # the overlap changes the cloudy columns


def test_maxrand_deletion_mutant_misses_arm_c(runs):
    fx, _, off, deleted = runs
    for key in deleted:
        np.testing.assert_array_equal(deleted[key], off[key], err_msg=key)
    assert columns_outside(fx, deleted, reference_arm="C").sum() >= 20
