"""NSSL 2-moment sedimentation (sediment1d + ziegfall1d + setvtz + fallout1d/calczgr1d/calcnfromz1d)
vs the pristine-WRF stage oracle: input stage S1 (state after calcnfromq) -> output stage S2
(state after sediment1d) and the surface-flux accumulators xfall, all 14 v034 cases.

fp64 port vs ``-fdefault-real-8`` oracle: |port-ref| <= 1e-10*|ref| + 1e-14*max|ref(var)| for every
species/level (measured: <= 1e-13 of the field scale, xfall <= 2e-16).
fp32 port vs WRF REAL oracle: band <= 5e-6 of the field scale (measured <= 2.5e-7: libm/pow ulps).
Mutants (deletion sensitivity, E39): rain fall-speed fit coefficient and the Mansell number
correction (calcnfromz1d) each break the fp64 gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_io as oio  # noqa: E402

from gpuwrf.physics.nssl2mom import fallspeed, sediment  # noqa: E402
from gpuwrf.physics.nssl2mom.constants import get_constants  # noqa: E402
from gpuwrf.physics.nssl2mom.indices import FP32, FP64  # noqa: E402

PREC = {"fp32": FP32, "fp64": FP64}


def run_case(mode, case):
    sp = oio.load_case(mode, case)
    an_in = oio.stage_an(sp, "S1")
    cols = {k: oio.stage_col(sp, "S0", k) for k in ("t0", "t7", "dn1", "dz2d")}
    an, xfall = sediment.sediment1d(an_in, cols["t0"], cols["t7"], cols["dn1"], cols["dz2d"],
                                    sp["scalars"]["DT"], get_constants(mode), PREC[mode])
    ref = oio.stage_an(sp, "S2")
    xref = np.array([0.0] + sp["stages"]["S2"]["xfall"])
    return np.asarray(an, np.float64), np.asarray(xfall, np.float64), ref, xref


def errors(an, xfall, ref, xref):
    """Return (max point-gate violation ratio fp64-style, max scale-relative error)."""
    worst_gate, worst_scale = 0.0, 0.0
    for il in range(1, ref.shape[0]):
        r, p = ref[il], an[il]
        scale = float(np.max(np.abs(r)))
        err = np.abs(p - r)
        tol = 1e-10 * np.abs(r) + 1e-14 * scale
        if scale == 0.0:
            assert float(np.max(err)) == 0.0, f"species {il}: nonzero port where oracle is zero"
            continue
        worst_gate = max(worst_gate, float(np.max(err / np.maximum(tol, 1e-300))))
        worst_scale = max(worst_scale, float(np.max(err)) / scale)
    xs = np.maximum(np.abs(xref), 1e-300)
    xe = np.abs(xfall - xref)
    xerr = float(np.max(np.where(np.abs(xref) > 0, xe / xs, xe)))
    return worst_gate, worst_scale, xerr


@pytest.mark.parametrize("case", oio.CASES)
def test_sediment_fp64_machine_precision(case):
    an, xfall, ref, xref = run_case("fp64", case)
    gate, scale, xerr = errors(an, xfall, ref, xref)
    assert gate <= 1.0, (case, gate, scale)
    assert xerr <= 1e-10, (case, xerr)


@pytest.mark.parametrize("case", oio.CASES)
def test_sediment_fp32_band(case):
    an, xfall, ref, xref = run_case("fp32", case)
    _gate, scale, xerr = errors(an, xfall, ref, xref)
    assert scale <= 5e-6, (case, scale)
    assert xerr <= 5e-6, (case, xerr)


def test_sediment_is_active():
    """The oracle stage must actually move mass (else parity is vacuous, E154)."""
    an, xfall, ref, xref = run_case("fp64", 14)
    s1 = oio.stage_an(oio.load_case("fp64", 14), "S1")
    for il in (4, 8, 11, 15, 17):  # qr qhl nr nhl vhl change in the hail warm-start case
        assert np.max(np.abs(ref[il] - s1[il])) > 0.0, il
    assert xref[4] > 0 and xref[8] > 0


def test_mutant_rain_fall_speed_fit(monkeypatch):
    monkeypatch.setattr(fallspeed, "_FRX", 516.0)
    an, xfall, ref, xref = run_case("fp64", 1)
    gate, _scale, xerr = errors(an, xfall, ref, xref)
    assert gate > 1.0 or xerr > 1e-10


def test_mutant_number_correction_removed(monkeypatch):
    monkeypatch.setattr(sediment, "calcnfromz1d",
                        lambda an, t0z, z0, db, t1, il, C, prec: an[fallspeed.LN[il]])
    an, xfall, ref, xref = run_case("fp64", 4)
    gate, _scale, _xerr = errors(an, xfall, ref, xref)
    assert gate > 1.0
