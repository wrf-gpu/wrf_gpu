"""NSSL 2-moment gs PART A (module lines 12620-16326) vs the pristine-WRF oracle G1 dump.

The G1 dump (all gs locals after the collection-efficiency loop, line 16326) is produced by the
print-only instrumented oracle (proofs/v034/oracle/nssl2mom, E26 bit-identical) and lives in the
lane work dir; tests skip loudly if it is absent.  Inputs: driver stage S2 (state after
sedimentation = gs entry) + S0 columns.
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

from gpuwrf.physics.nssl2mom import constants, gs_a, indices  # noqa: E402

# scratch temporaries reassigned in part B before use, dead-config locals, or Fortran scalars
# that hold the value of the LAST gathered point (per-point here)
NOT_COMPARED = {"tqvcon"}
REL = {"fp64": 1e-10, "fp32": 2e-6}

_have = (oio.ORACLE_WORK / "gs_vars_dims.txt").exists()
pytestmark = pytest.mark.skipif(not _have, reason=f"gs dumps missing in {oio.ORACLE_WORK} (dev-only oracle)")

SETVTZ_OUT = ("vtxbar", "xdia", "cx", "xv", "xmas", "axx", "bxx", "xdn", "cdxgs")


def _inputs(mode, case):
    sp = oio.load_case(mode, case)
    an = oio.stage_an(sp, "S2")
    cols = {n: oio.stage_col(sp, "S0", n) for n in ("t0", "t7", "t00", "t77", "pn", "wn", "dn1", "dz2d")}
    return an, cols, sp["scalars"]["DT"]


def _injector(G1):
    """setvtz replacement that returns the oracle's post-setvtz values (testing the rest of A)."""
    idx = G1["kgs"][: G1["ngscnt"]] - 1

    def fn(G, C, prec):
        out = {}
        for k in SETVTZ_OUT:
            ref = G1[k]
            if isinstance(ref, dict):
                base = G.get(k, {})
                d = {}
                for kk, v in ref.items():
                    b = np.array(base[kk]) if kk in base else np.zeros(G["temg"].shape)
                    b = b.astype(np.float64).copy()
                    b[idx] = v
                    d[kk] = jax.numpy.asarray(b, prec.R)
                out[k] = d
        return out

    return fn


def run_case(mode, case, inject):
    prec = indices.FP32 if mode == "fp32" else indices.FP64
    C = constants.get_constants(mode)
    an, c, dt = _inputs(mode, case)
    G1 = oio.load_gs_dump(mode, case, "G1")
    G = gs_a.gs_part_a(an, c["t0"], c["t7"], c["t00"], c["t77"], c["pn"], c["wn"], c["dn1"], c["dz2d"], dt,
                       C, prec, setvtz_fn=_injector(G1) if inject else None)
    return G, G1


def _cmp(name, mine, ref, idx, rel, fails, worst):
    if isinstance(ref, dict):
        for k, v in ref.items():
            if k not in mine:
                continue
            _cmp(f"{name}[{k}]", mine[k], v, idx, rel, fails, worst)
        return
    if isinstance(ref, np.ndarray) and isinstance(mine, dict):
        mine = np.array([np.asarray(mine[k]) for k in sorted(mine)])
        ref = ref[: len(mine)]
    m = np.asarray(mine)
    r = np.asarray(ref)
    if m.ndim >= 1 and r.ndim == 1 and m.shape[-1] != r.shape[0] and m.shape[-1] > r.shape[0]:
        m = m[..., idx]
    elif m.ndim >= 1 and r.ndim == 0:
        return  # per-point here, scalar in Fortran
    if r.dtype == bool or m.dtype == bool:
        if not np.array_equal(m.astype(bool), r.astype(bool)):
            fails.append(f"{name}: {m} != {r}")
        return
    m = m.astype(np.float64)
    r = r.astype(np.float64)
    err = np.abs(m - r)
    tol = rel * np.abs(r) + 1e-300
    bad = err > tol
    w = float(np.max(np.where(np.abs(r) > 0, err / np.maximum(np.abs(r), 1e-300), err))) if err.size else 0.0
    worst[name] = w
    if np.any(bad):
        i = int(np.argmax(np.where(bad, err / np.maximum(np.abs(r), 1e-300), 0)))
        fails.append(f"{name}: max rel {w:.3e} at {i}: mine {m.ravel()[i]!r} ref {r.ravel()[i]!r}")


def check(mode, case, inject):
    G, G1 = run_case(mode, case, inject)
    idx = G1["kgs"][: G1["ngscnt"]] - 1
    gathered = np.asarray(G["gathered"])
    assert np.array_equal(np.nonzero(gathered)[0], idx), (np.nonzero(gathered)[0], idx)
    fails, worst = [], {}
    has = {il: np.asarray(G["qx"][il])[idx] > float(C_of(mode).qxmin[il]) for il in range(3, 9)}
    for k in gs_a.A_TO_B_KEYS:
        if k in NOT_COMPARED or k not in G1:
            continue
        assert k in G, f"part A did not produce {k}"
        mine, ref = G[k], G1[k]
        if k in MASKED:  # set by setvtz only where the species is present; garbage elsewhere in Fortran
            mine = {il: np.where(has[il], np.asarray(mine[il])[idx], 0.0) for il in MASKED[k]}
            ref = {il: np.where(has[il], ref[il], 0.0) for il in MASKED[k]}
        _cmp(k, mine, ref, idx, REL[mode], fails, worst)
    return fails, worst


def C_of(mode):
    return constants.get_constants(mode)


# gs locals assigned by setvtz only for present graupel/hail (cdxgs, axx, bxx: lines 7299-7453)
MASKED = {"cdxgs": (7, 8), "axx": (7, 8), "bxx": (7, 8)}


@pytest.mark.parametrize("case", oio.CASES)
@pytest.mark.parametrize("mode", ["fp64", "fp32"])
def test_gs_part_a(mode, case):
    fails, worst = check(mode, case, inject=False)
    assert not fails, "\n".join(fails[:20])


def test_gs_part_a_mutant_sensitive(monkeypatch):
    """Mutants must fail the gate: (1) graupel-ice sticking coefficient eii0 x 1.0001,
    (2) dropping the infdo<2 Z-weighted rain fall-speed assignment (setvtz lines 7075-7079)."""
    C = constants.get_constants("fp64")
    import types
    C2 = types.SimpleNamespace(**vars(C))
    C2.eii0 = C.eii0 * 1.0001
    monkeypatch.setattr(constants, "get_constants", lambda mode: C2)
    G, G1 = run_case("fp64", 4, inject=False)
    fails, worst = [], {}
    _cmp("ehi", G["ehi"], G1["ehi"], G1["kgs"][: G1["ngscnt"]] - 1, 1e-10, fails, worst)
    assert fails, "mutant eii0 not detected"
    monkeypatch.setattr(constants, "get_constants", lambda mode: C)
    G, G1 = run_case("fp64", 1, inject=False)
    G["vtxbar"] = dict(G["vtxbar"])
    G["vtxbar"][(4, 3)] = G["vtxbar"][(4, 3)] * 0.0
    fails = []
    _cmp("vtxbar", G["vtxbar"], G1["vtxbar"], G1["kgs"][: G1["ngscnt"]] - 1, 1e-10, fails, worst)
    assert fails, "mutant vtxbar(lr,3) not detected"
