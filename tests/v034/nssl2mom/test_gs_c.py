"""nssl_2mom_gs PART C (lines 19696-25151) vs the pristine-WRF oracle, all 14 v034 cases.

Inputs: the gs locals dumped at line 19695 (G2, print-only instrumented oracle, E26
bit-identical) expanded to the full column, S2 ``an`` and S0 ``t0/t7/dz2d``.  Reference: S3
``an`` (every species, every level) and S3 ``t0``.  The G2 dumps are DEV assets in the lane
directory (``oracle_io.ORACLE_WORK``); the test skips loudly when they are absent.

Gate: fp64 port vs fp64 oracle <= 1e-10 relative (with a 1e-14*max|ref| floor per species);
fp32 port vs fp32 oracle within a REAL band.  Deletion-sensitive: a halved rime-density coefficient
(ice/snow -> graupel) and a dropped rain-evaporation term must break the fp64 comparison.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import oracle_io as o  # noqa: E402
from gpuwrf.physics.nssl2mom import gs_c  # noqa: E402
from gpuwrf.physics.nssl2mom.constants import get_constants  # noqa: E402
from gpuwrf.physics.nssl2mom.indices import FP32, FP64, NA  # noqa: E402

_HAVE_DUMPS = (o.ORACLE_WORK / "gs_vars_dims.txt").exists() and (
    o.ORACLE_WORK / "build_fp64_instr" / "case_1.txt").exists()
pytestmark = pytest.mark.skipif(
    not _HAVE_DUMPS, reason=f"gs-internal oracle dumps missing under {o.ORACLE_WORK} (run "
    "proofs/v034/oracle/nssl2mom/build_and_run.sh)")

FP64_RTOL = 1e-10
FP32_RTOL = 5e-3  # REAL band (documented in the report); fp64 is the parity gate


def _full_column_G(g, nz, R):
    n = g["ngscnt"]
    k = np.asarray(g["kgs"])[:n] - 1
    gathered = np.zeros(nz, bool)
    gathered[k] = True

    def expand(a):
        a = np.asarray(a)
        out = np.full(nz, a[0], dtype=a.dtype)
        out[k] = a
        if out.dtype.kind == "f":
            out = out.astype(R)
        return out

    G = {}
    for name, v in g.items():
        if name == "ngscnt":
            continue
        if isinstance(v, dict):
            G[name] = {kk: expand(vv) for kk, vv in v.items()}
        elif isinstance(v, np.ndarray) and v.shape == (n,):
            G[name] = expand(v)
        elif isinstance(v, float):
            G[name] = R(v)
        else:
            G[name] = v
    G["cautn"] = expand(g["cautn"]).astype(np.float64)  # DOUBLE PRECISION local
    kk = np.arange(1, nz + 1)
    G["kgs"] = kk
    G["kgsp"] = np.minimum(kk + 1, nz - 1)
    G["kgsm"] = np.maximum(kk - 1, 1)
    G["gathered"] = gathered
    return G


def _run(mode, case):
    prec = FP64 if mode == "fp64" else FP32
    R = prec.nR
    C = get_constants(mode)
    sp = o.load_case(mode, case)
    nz = sp["scalars"]["KX"]
    G = _full_column_G(o.load_gs_dump(mode, case, "G2"), nz, R)
    G["gz"] = o.stage_col(sp, "S0", "dz2d").astype(R)
    an = o.stage_an(sp, "S2").astype(R)
    t0 = o.stage_col(sp, "S0", "t0").astype(R)
    t7 = o.stage_col(sp, "S0", "t7").astype(R)
    z = np.zeros(nz, R)
    an_new, ts, st = gs_c.gs_part_c(G, an, t0, z, z, z, z, z, z, t7, z, z, C, prec,
                                    sp["scalars"]["DT"], return_state=True)
    return np.asarray(an_new, np.float64), [np.asarray(t, np.float64) for t in ts], st, sp


def _max_rel(port, ref):
    floor = 1e-14 * (np.max(np.abs(ref)) + 1e-300)
    return float(np.max(np.abs(port - ref) / (np.abs(ref) + floor)))


def _errors(an_new, ts, sp):
    ref = o.stage_an(sp, "S3")
    errs = {il: _max_rel(an_new[il], ref[il]) for il in range(1, NA + 1)}
    for i in range(10):
        r = o.stage_col(sp, "S3", f"t{i}")
        errs[f"t{i}"] = _max_rel(ts[i], r)
    return errs


@pytest.mark.parametrize("case", o.CASES)
def test_gs_part_c_fp64_parity(case):
    an_new, ts, st, sp = _run("fp64", case)
    errs = _errors(an_new, ts, sp)
    bad = {k: v for k, v in errs.items() if not v <= FP64_RTOL}
    assert not bad, f"case {case}: fp64 rel errors above {FP64_RTOL}: {bad}"
    assert st["consumed"] <= set(gs_c.G2_KEYS_CONSUMED), sorted(st["consumed"] - set(gs_c.G2_KEYS_CONSUMED))


@pytest.mark.parametrize("case", o.CASES)
def test_gs_part_c_fp32_band(case):
    an_new, ts, _, sp = _run("fp32", case)
    errs = _errors(an_new, ts, sp)
    bad = {k: v for k, v in errs.items() if not v <= FP32_RTOL}
    assert not bad, f"case {case}: fp32 rel errors above {FP32_RTOL}: {bad}"


def _run_with(mode, case, C=None, mutate_G=None):
    prec = FP64 if mode == "fp64" else FP32
    R = prec.nR
    C = C or get_constants(mode)
    sp = o.load_case(mode, case)
    nz = sp["scalars"]["KX"]
    G = _full_column_G(o.load_gs_dump(mode, case, "G2"), nz, R)
    G["gz"] = o.stage_col(sp, "S0", "dz2d").astype(R)
    if mutate_G:
        mutate_G(G)
    z = np.zeros(nz, R)
    an_new, ts = gs_c.gs_part_c(G, o.stage_an(sp, "S2").astype(R), o.stage_col(sp, "S0", "t0").astype(R),
                                z, z, z, z, z, z, o.stage_col(sp, "S0", "t7").astype(R), z, z, C, prec,
                                sp["scalars"]["DT"])
    return _errors(np.asarray(an_new, np.float64), [np.asarray(t, np.float64) for t in ts], sp)


def test_gs_part_c_mutants_are_caught():
    """Deletion sensitivity on processes ACTIVE in the oracle cases: rime density of the
    ice/snow -> graupel conversions (rimc1, case 4) and rain evaporation (fvce, case 12).
    NB the graupel -> hail conversion branch (qxd1 > 10*qxmin(lhl)) is reachable but inactive in
    all 14 cases, so a gaminterp mutant is NOT detectable (coverage gap, reported)."""
    C = get_constants("fp64")
    Cm = types.SimpleNamespace(**vars(C))
    Cm.rimc1 = C.rimc1 * 0.5
    assert max(_run_with("fp64", 4, C=Cm).values()) > 1e-6

    def drop_evap(G):
        G["fvce"] = 0.0 * G["fvce"]

    assert max(_run_with("fp64", 12, mutate_G=drop_evap).values()) > 1e-6
