"""End-to-end NSSL mp=18 column (nssl_2mom_driver) vs the pristine-WRF oracle: WRF inputs (*_IN) ->
every prognostic output (*_OUT), surface precipitation, dbz and effective radii; 14 cases, fp64 and
fp32 builds.  This is the graduation parity gate of the port (stage tests localise failures)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_io as oio  # noqa: E402
from test_driver_io import IN_NAMES, fields_in  # noqa: E402

from gpuwrf.physics.nssl2mom import driver  # noqa: E402
from gpuwrf.physics.nssl2mom.constants import get_constants  # noqa: E402
from gpuwrf.physics.nssl2mom.indices import FP32, FP64  # noqa: E402

PREC = {"fp32": FP32, "fp64": FP64}


_JIT = {}


def _column_fn(mode, dt):
    """One jitted column program per (build, dt); WRF's itimestep==1 branch is a traced flag."""
    key = (mode, float(dt))
    if key not in _JIT:
        C, prec = get_constants(mode), PREC[mode]
        _JIT[key] = jax.jit(lambda f, first: driver.nssl2mom_column(f, float(dt), C, prec, first_step=first,
                                                                     diag=True))
    return _JIT[key]


def run_case(mode, case):
    sp = oio.load_case(mode, case)
    fn = _column_fn(mode, sp["scalars"]["DT"])
    fields = {k: np.asarray(v, PREC[mode].nR) for k, v in fields_in(sp).items()}
    out = fn(fields, np.bool_(sp["scalars"]["ITIMESTEP"] == 1))
    return sp, jax.tree_util.tree_map(np.asarray, out)


def errs(a, ref, floor):
    """(max relative error over |ref| > floor, max abs error) -- floor = physically-zero level."""
    a = np.asarray(a, np.float64)
    ref = np.asarray(ref, np.float64)
    d = np.abs(a - ref)
    big = np.abs(ref) > floor
    rel = float(np.max(d[big] / np.abs(ref[big]))) if big.any() else 0.0
    small = float(np.max(d[~big])) if (~big).any() else 0.0
    return rel, small


# physically-zero floors (below every scheme threshold: qxmin 1e-13..1e-12 kg/kg, cxmin 1e-8 #/m3)
FLOOR = {k: 1e-14 for k in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")}
FLOOR.update({k: 1e-6 for k in ("qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn")})
FLOOR.update(th=0.0, qvolg=1e-17, qvolh=1e-17)


@pytest.mark.parametrize("case", oio.CASES)
def test_e2e_fp32_dual_reference(case):
    """fp32 port vs WRF REAL build, dual-reference band (v023 README section 4): fp32 threshold/onset
    cells may flip differently in the two REAL builds (e.g. case 5 fully-sublimating snow cell: WRF-fp32
    1.8e-12, WRF-fp64 0; case 8 ice: port == WRF-fp64 to 2.5e-6 while WRF-fp32 differs 4.6e-4), so each
    value must match the fp32 OR the fp64 oracle within rtol, plus an absolute floor of 1e-6 x the
    column maximum (physically zero)."""
    sp, (out, precip, diag) = run_case("fp32", case)
    sp64 = oio.load_case("fp64", case)
    rtol = 2e-3
    mass_of = {"qndrop": "qc", "qnr": "qr", "qni": "qi", "qns": "qs", "qng": "qg", "qnh": "qh",
               "qvolg": "qg", "qvolh": "qh"}
    for k, v in IN_NAMES.items():
        p = np.asarray(out[k], np.float64)
        r32 = np.asarray(sp["columns"][v + "_OUT"], np.float64)
        r64 = np.asarray(sp64["columns"][v + "_OUT"], np.float64)
        assert np.all(np.isfinite(p)), k
        atol = 1e-6 * max(np.max(np.abs(r32)), np.max(np.abs(r64)), 1e-30)
        ok32 = np.abs(p - r32) <= rtol * np.abs(r32) + atol
        ok64 = np.abs(p - r64) <= rtol * np.abs(r64) + atol
        bad = ~(ok32 | ok64)
        if k in mass_of:
            # number/volume of a category whose MASS is physically zero (< 1e-10 kg/kg) in the port and
            # both references (fp32 sublimation dust, e.g. case 5/12 snow: WRF-fp32 keeps 1.8e-12 kg/kg with
            # ~30 #/kg, WRF-fp64 zeroes it) is not gated; the mass itself is gated above.
            mk = mass_of[k]
            mv = IN_NAMES[mk]
            mass = np.maximum.reduce([np.abs(np.asarray(out[mk], np.float64)),
                                      np.abs(np.asarray(sp["columns"][mv + "_OUT"], np.float64)),
                                      np.abs(np.asarray(sp64["columns"][mv + "_OUT"], np.float64))])
            bad &= mass >= 1e-10
        assert not bad.any(), (k, np.nonzero(bad)[0][:5], p[bad][:3], r32[bad][:3], r64[bad][:3])
    for k in ("rainncv", "snowncv", "grplncv", "hailncv"):
        r32, r64, p = sp["scalars"][k.upper()], sp64["scalars"][k.upper()], float(precip[k])
        assert min(abs(p - r32), abs(p - r64)) <= rtol * max(abs(r32), abs(r64)) + 1e-12, (k, p, r32, r64)


# Machine-precision knife-edge in WRF itself (proven, proofs/v034/oracle/nssl2mom/knife_edge): nssl_2mom_gs
# line 24553 zeroes the droplet number only if the updated cloud mass is EXACTLY <= 0.  Case 15 k=12 (-37.5 C)
# freezes all cloud water; a 1-ulp change of the WRF input (eps ~2e-16) flips pristine WRF's own final activated
# CCN between 1.762593549821861e7 (stored oracle) and 1.550161978460388e7.  The port must equal one of the two.
KNIFE_EDGE = {15: {"levels": (12,), "vars": ("qnn",), "wrf_alt": {"qnn": 1.550161978460388e7}}}


@pytest.mark.parametrize("case", oio.CASES)
def test_e2e_fp64(case):
    sp, (out, precip, diag) = run_case("fp64", case)
    col = sp["columns"]
    knife = KNIFE_EDGE.get(case)
    for k, v in IN_NAMES.items():
        assert np.all(np.isfinite(np.asarray(out[k]))), k  # E200: NaN/Inf never pass
        p = np.asarray(out[k], np.float64)
        ref = np.asarray(col[v + "_OUT"], np.float64)
        # relative to the value, with a floor of 1e-13 x the column operand scale (input/output max): a
        # near-total conversion leaves a cancellation residual (case 15: rain 8e-4 -> 1e-8 after freezing)
        scale = max(np.max(np.abs(ref)), np.max(np.abs(np.asarray(col[v + "_IN"], np.float64))))
        bad = np.abs(p - ref) > 1e-10 * np.abs(ref) + 1e-13 * scale
        if knife and k in knife["vars"]:
            for lev in knife["levels"]:
                alt = knife["wrf_alt"][k]
                assert (not bad[lev]) or abs(p[lev] - alt) <= 1e-12 * abs(alt), (k, lev, p[lev], ref[lev], alt)
                bad[lev] = False
        assert not bad.any(), (k, np.nonzero(bad)[0][:5], p[bad][:3], ref[bad][:3])
    if knife:
        qc_in = np.max(np.abs(np.asarray(col["QC_IN"], np.float64)))
        for lev in knife["levels"]:
            assert abs(float(out["qc"][lev]) - col["QC_OUT"][lev]) <= 1e-15 * qc_in
    for k in ("rainncv", "snowncv", "grplncv", "hailncv"):
        ref = sp["scalars"][k.upper()]
        assert abs(float(precip[k]) - ref) <= 1e-9 * abs(ref) + 1e-15, (k, float(precip[k]), ref)
    for k, ref in (("dbz", "DBZ_OUT"), ("re_cloud", "RE_CLOUD_OUT"), ("re_ice", "RE_ICE_OUT"), ("re_snow", "RE_SNOW_OUT")):
        rel, small = errs(diag[k], col[ref], 0.0)
        assert rel <= 1e-10, (k, rel)


def test_e2e_mutant_is_caught():
    """E39: a perturbed process coefficient inside gs (rime density constant) breaks fp64 parity."""
    from gpuwrf.physics.nssl2mom import gs_c
    sp = oio.load_case("fp64", 4)
    C = get_constants("fp64")
    import dataclasses  # noqa: F401
    from types import SimpleNamespace
    Cm = SimpleNamespace(**{**vars(C), "rimc1": C.rimc1 * 0.5})
    fields = {k: np.asarray(v, np.float64) for k, v in fields_in(sp).items()}
    fn = jax.jit(lambda f: driver.nssl2mom_column(f, float(sp["scalars"]["DT"]), Cm, FP64, first_step=True, diag=False))
    out, _p, _d = jax.tree_util.tree_map(np.asarray, fn(fields))
    rel, _ = errs(out["qg"], sp["columns"]["QH_OUT"], FLOOR["qg"])
    assert rel > 1e-6, rel
