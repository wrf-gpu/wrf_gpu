"""nssl_2mom_gs part B (lines 16327-19695) vs the pristine-WRF oracle gs-internal dumps G1 -> G2.

DEV-level gate: needs the lane oracle work dir (``oracle_io.ORACLE_WORK``) holding the instrumented
oracle text dumps; skips loudly (reason printed) when it is absent.
"""

from __future__ import annotations

import os
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(__file__))
import oracle_io  # noqa: E402

from gpuwrf.physics.nssl2mom import gs_b  # noqa: E402
from gpuwrf.physics.nssl2mom.constants import get_constants  # noqa: E402
from gpuwrf.physics.nssl2mom.indices import FP32, FP64  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (oracle_io.ORACLE_WORK / "build_fp64_instr" / "case_1.txt").exists(),
    reason=f"gs-internal oracle dumps absent under {oracle_io.ORACLE_WORK}",
)

# arrays part B assigns that part C (lines 19696-25151) reads (grep of the pristine source)
COMPARED = (
    "chaci chaci0 chacr chacs chacs0 chacw chlaci chlaci0 chlacr chlacs chlacs0 chlacw chlmlr chlmlrr chmlr "
    "chmlrr ciacr ciacrf ciacrs ciacw cicichr ciihr cimlr craci cracr cracw crcnw crfrz crfrzf crfrzs csaci "
    "csacr csacs csacw cscnvi cscnvis csmlr csmlrr csplinter csplinter2 cwctfz cwctfzc cwctfzp cwfrz cwfrzc "
    "cwfrzp dqci dqcw dqvcnd dqwv fraci fracl fvce fwet1 fwet2 hlvent hlventy hwvent hwventy qhaci qhaci0 "
    "qhacr qhacs qhacs0 qhacw qhdsv qhfzh qhlaci qhlaci0 qhlacr qhlacs qhlacs0 qhlacw qhldsv qhlfzhl qhlmlr "
    "qhmlr qiacr qiacrf qiacrs qiacw qicichr qidsv qiihr qimlr qipiphr qitmp qraci qracs qracw qrcnw qrfrz "
    "qrfrzf qrfrzs qsaci qsacr qsacw qscnvi qsdsv qsimxdep qsimxsub qsmlr qsplinter qsplinter2 qss qwcnr "
    "qwctfz qwctfzc qwfrz qwfrzc qwfrzp raindn rimdn rwcap rwvent rwventz swcap swvent vhacr vhacw vhfzh "
    "vhlacr vhlacw vhlfzhl vhlsoak vhsoak viacrf vrfrzf vsacw vtxbar xcolmn xplate zhacr zhacw zhlmlr "
    "zhlmlrr zhlshr zhlshrr zhmlr zhmlrr zhshr zhshrr zrcnw zrfrz zrfrzf zsmlrr "
    # B-only diagnostics that are cheap to check as well
    "cautn ec0 fvent civent fmlt1 fmlt2 fvds cicap hwcap hlcap qctmp qrtmp dqcitmp"
).split()
NOT_PRODUCED_HERE = {"rwventz"}  # ipconc < 7: never assigned in part B for mp=18 (stays stale)


def _typed(name, val, prec):
    import jax.numpy as jnp

    typ = oracle_io._gs_decls()[name][0] if name in oracle_io._gs_decls() else "real"
    dt = {"real": prec.R, "doubleprecision": jnp.float64, "integer": jnp.int32}.get(typ, bool)
    conv = lambda v: jnp.asarray(v, dt)  # noqa: E731
    if isinstance(val, dict):
        return {k: conv(v) for k, v in val.items()}
    if isinstance(val, np.ndarray):
        return conv(val)
    return val


def build_g(mode, case):
    prec = FP64 if mode == "fp64" else FP32
    raw = oracle_io.load_gs_dump(mode, case, "G1")
    G = {k: _typed(k, v, prec) for k, v in raw.items() if k not in ("ngscnt", "kgs")}
    sp = oracle_io.load_case(mode, case)
    dz = np.asarray(sp["stages"]["S0"]["dz2d"])
    import jax.numpy as jnp

    kgs = np.asarray(raw["kgs"])[: raw["ngscnt"]]  # the generic dump re-writes kgs(1:ngs)
    G["gz"] = jnp.asarray(dz[kgs - 1], prec.R)
    return G, prec, sp["scalars"]["DT"]


NPAD = 64  # ngs: pad every case to one shape so the jitted part B compiles once per mode


def _pad(v, n):
    import jax.numpy as jnp

    if isinstance(v, dict):
        return {k: _pad(x, n) for k, x in v.items()}
    if hasattr(v, "shape") and v.ndim == 1 and v.shape[0] == n:
        return jnp.concatenate([v, jnp.repeat(v[-1:], NPAD - n)])
    return v


def _cut(v, n):
    if isinstance(v, dict):
        return {k: _cut(x, n) for k, x in v.items()}
    if hasattr(v, "shape") and v.ndim == 1 and v.shape[0] == NPAD:
        return v[:n]
    return v


_JIT = {}


def run_case(mode, case, jit=True, fresh=False):
    G, prec, dt = build_g(mode, case)
    C = get_constants(mode)
    ref = oracle_io.load_gs_dump(mode, case, "G2")
    if not jit:
        return gs_b.gs_part_b(G, C, prec, dt), ref
    n = int(ref["ngscnt"])
    if mode not in _JIT or fresh:
        _JIT[mode] = jax.jit(lambda g, d: gs_b.gs_part_b(g, C, prec, d))
    scal = {k: v for k, v in G.items() if not isinstance(v, (dict,)) and not hasattr(v, "shape")}
    arrs = {k: _pad(v, n) for k, v in G.items() if k not in scal}
    out = _JIT[mode](arrs | {k: np.float64(v) if isinstance(v, float) else v for k, v in scal.items()}, dt)
    return _cut(out, n), ref


def _flat(v):
    if isinstance(v, dict):
        return {k: np.asarray(x, np.float64) for k, x in v.items()}
    return {None: np.asarray(v, np.float64)}


# Variables whose value is a small difference of large operands: the error is measured against the
# operand scale S instead of |ref| (conditioning, not a port difference):
#  * dqwv/dqvcnd/dqci = qv - qis(T) near ice saturation, S = max qv of the column (libm exp ulps);
#  * qitmp/qsimxsub/qsimxdep: ice mass after (near-)total sublimation, S = max initial ice mass;
#  * vhsoak/vhlsoak: (1 - xdn/xdnmx) with xdn == xdnmx - O(ulp) (values ~1e-24, physically 0);
#    eager evaluation is bit-exact, jit differs only by XLA:CPU's x/const -> x*(1/const) rewrite (E130);
#  * qiacrs/ciacrs = (1 - frach)*qiacr/ciacrf with frach = 0.5*(1+tanh(x)) (ibiggsnow=3 small frozen drops
#    to snow): for x >~ 10 this is 1 - tanh(x) cancellation; XLA's tanh differs from glibc's by ~1e-16
#    absolute (NumPy/glibc reproduces the oracle exactly, case 15: 1.9e-6 rel at qiacrs/qiacr = 6e-11),
#    so S = column max of the operand qiacr/ciacrf.
CONDITIONED = {
    "dqwv": "qv", "dqvcnd": "qv", "dqci": "qv",
    "qitmp": "ice", "qsimxsub": "ice", "qsimxdep": "ice",
    "vhsoak": "vsoak_h", "vhlsoak": "vsoak_hl",
    "qiacrs": "qiacr", "ciacrs": "ciacr",
}


def _scales(g1, ref):
    qx = g1["qx"]
    ice = np.asarray(qx[5]) + np.asarray(qx[6]) + np.asarray(qx[7]) + np.asarray(qx[8])
    rho0 = np.asarray(g1["rho0"])
    return {
        "qv": float(np.max(np.abs(qx[2]))),
        "ice": float(np.max(ice)),
        "vsoak_h": float(np.max(rho0 * np.abs(np.asarray(ref["qhmlr"])) / 900.0)) or 1.0,
        "vsoak_hl": float(np.max(rho0 * np.abs(np.asarray(ref["qhlmlr"])) / 900.0)) or 1.0,
        "qiacr": float(np.max(np.abs(np.asarray(ref["qiacr"])))) or 1.0,
        "ciacr": float(np.max(np.abs(np.asarray(ref["ciacr"])))) or 1.0,
    }


def compare(out, ref, names=COMPARED, g1=None):
    """Per variable max error: |port-ref| / max(|ref|, 1e-12*max|ref|), or |port-ref|/S for CONDITIONED."""
    sc = _scales(g1, ref) if g1 is not None else {}
    res = {}
    for n in names:
        if n in NOT_PRODUCED_HERE:
            continue
        p, q = _flat(out[n]), _flat(ref[n])
        worst = 0.0
        for k in q:
            a, b = p[k], q[k]
            if n in CONDITIONED and g1 is not None and sc[CONDITIONED[n]] > 0.0:
                den = np.full(b.shape, sc[CONDITIONED[n]])
            else:
                den = np.maximum(np.abs(b), 1e-12 * max(float(np.max(np.abs(b))) if b.size else 0.0, 1e-300))
            with np.errstate(invalid="ignore", divide="ignore"):
                err = np.where(a == b, 0.0, np.abs(a - b) / den)
            err = np.where(np.isfinite(err), err, np.inf)  # NaN/Inf mismatches never pass (E200)
            if err.size:
                worst = max(worst, float(np.max(err)))
        res[n] = worst
    return res


@pytest.mark.parametrize("case", oracle_io.CASES)
def test_gs_b_fp64_matches_wrf(case):
    out, ref = run_case("fp64", case)
    errs = compare(out, ref, g1=oracle_io.load_gs_dump("fp64", case, "G1"))
    bad = {k: v for k, v in errs.items() if not v <= 1e-10}
    assert not bad, bad


@pytest.mark.parametrize("case", oracle_io.CASES)
def test_gs_b_fp32_band(case):
    """fp32 port vs fp32 (WRF REAL) oracle: band, not bitwise (XLA vs glibc libm, fusion order)."""
    out, ref = run_case("fp32", case)
    errs = compare(out, ref, g1=oracle_io.load_gs_dump("fp32", case, "G1"))
    bad = {k: v for k, v in errs.items() if not v <= 1e-4}
    assert not bad, bad


def test_gs_b_mutant_detected(monkeypatch):
    """Deletion sensitivity: perturbing the rain-graupel collection kernel must break parity."""
    orig = gs_b._AA2
    monkeypatch.setattr(gs_b, "_AA2", orig * (1.0 + 1e-6))
    out, ref = run_case("fp64", 4, fresh=True)
    _JIT.pop("fp64", None)  # never reuse the mutant trace
    errs = compare(out, ref, names=("qracw", "qraci", "cracw"))
    assert max(errs.values()) > 1e-8, errs
