"""NSSL 2-moment NUCOND/QVEXCESS port vs the pristine WRF oracle (CPU only).

Gates
1. Driver-stage parity, all 14 v034 savepoint cases: S3 (after nssl_2mom_gs) -> S4 (after NUCOND),
   every species/level + t0 + ssat.  fp64 port vs ``-fdefault-real-8`` oracle <= 1e-10 relative;
   fp32 port vs WRF REAL oracle reported as a band (jitted XLA vs gfortran/glibc, no bitwise claim).
   ``ssat`` = 100*(qv/qvs - 1) [%] is gated in % units (floor 1 %) because it is a cancellation
   quantity (1 ulp of qv/qvs near saturation is 1e-9 relative of ssat).
2. Adversarial parity: randomized + targeted columns run through the PRISTINE NUCOND called directly
   (dev oracle = pristine module + ``public nucond`` only), dataset ``nucond_adversarial.npz``.
   Covers branches the 14 cases never reach: RK2c time-step halving retry, maxsupersat cap
   (QVEXCESS at label 631), near-saturation ``delta = 0.1*dtp``, droplet/rain size clamps,
   cx<=cxmin re-seeding, full and partial cloud evaporation, nucleation with/without CCN depletion.
3. Mutants (deletion sensitivity, E39): dropping the maxsupersat QVEXCESS cap or the condensation
   loop must break gate 2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_io as o  # noqa: E402

from gpuwrf.physics.nssl2mom import nucond as nucond_mod  # noqa: E402
from gpuwrf.physics.nssl2mom.constants import get_constants  # noqa: E402
from gpuwrf.physics.nssl2mom.indices import Prec  # noqa: E402

_ADV_CANDIDATES = (
    o.SAVEPOINTS / "nucond_adversarial.npz",
    Path(os.environ.get("NSSL_O1_NUCOND_ADV", "<USER_HOME>/wrf_gpu2_lanes/o1-nssl/nucond_dev/nucond_adversarial.npz")),
)
TOL64 = 1e-10
# fp32: jitted XLA:CPU (FMA contraction, E107) vs gfortran -O2 + glibc; measured 6.7e-5 state / 1.2e-4 ssat
TOL32 = {"state": 2e-4, "ssat": 5e-4}


def _relerr(port, ref):
    port = np.asarray(port, np.float64)
    ref = np.asarray(ref, np.float64)
    scale = np.max(np.abs(ref)) if ref.size else 0.0
    if scale == 0.0:
        return float(np.max(np.abs(port))) if port.size else 0.0
    return float(np.max(np.abs(port - ref) / np.maximum(np.abs(ref), 1e-12 * scale)))


def _ssat_err(port, ref):
    """ssat = 100*(qv/qvs - 1) [%] is a cancellation quantity: gate it in % units (floor 1 %)."""
    port = np.asarray(port, np.float64)
    ref = np.asarray(ref, np.float64)
    return float(np.max(np.abs(port - ref) / np.maximum(np.abs(ref), 1.0)))


_JIT = {}


def _run(mode, an, dn, t77, pn, w, dt):
    C = get_constants(mode)
    P = Prec(mode)
    R = P.nR
    key = (mode, id(nucond_mod.qvexcess), id(nucond_mod.lax.while_loop))
    if key not in _JIT:  # jit like the product; re-trace when a mutant patches the module
        _JIT[key] = jax.jit(lambda a, d, t, p, ww, dtp: nucond_mod.nucond(a, d, t, p, ww, dtp, C, P))
    return _JIT[key](jnp.asarray(np.asarray(an, R)), jnp.asarray(np.asarray(dn, R)),
                     jnp.asarray(np.asarray(t77, R)), jnp.asarray(np.asarray(pn, R)),
                     jnp.asarray(np.asarray(w, R)), jnp.asarray(dt, R))


def _stage_errors(mode):
    worst = {}
    for c in o.CASES:
        sp = o.load_case(mode, c)
        cols = [o.stage_col(sp, "S0", k) for k in ("dn1", "t77", "pn", "wn")]
        an2, t0, ss = _run(mode, o.stage_an(sp, "S3"), *cols, sp["scalars"]["DT"])
        ref = o.stage_an(sp, "S4")
        an2 = np.asarray(an2, np.float64)
        for il in range(1, o.NA + 1):
            worst[il] = max(worst.get(il, 0.0), _relerr(an2[il], ref[il]))
        worst["t0"] = max(worst.get("t0", 0.0), _relerr(t0, o.stage_col(sp, "S4", "t0")))
        worst["ssat"] = max(worst.get("ssat", 0.0), _ssat_err(ss, o.stage_col(sp, "S4", "ssat")))
        # NUCOND never writes t9 (driver passes it, the routine ignores it) -> not an argument of the port
        assert np.array_equal(o.stage_col(sp, "S4", "t9"), o.stage_col(sp, "S3", "t9"))
    return worst


def _adv_path():
    for p in _ADV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"NUCOND adversarial oracle dataset missing; looked in {_ADV_CANDIDATES}")


def _adv_errors(mode):
    d = np.load(_adv_path())
    in_sp, out_sp = list(d["in_species"]), list(d["out_species"])
    sets = sorted({k.split("/")[0] for k in d.files if "/" in k})
    worst = {}
    for s in sets:
        p = f"{s}/{mode}/"
        a_in = d[p + "an_in"]
        an = np.zeros((o.NA + 1,) + a_in.shape[1:])
        an[in_sp] = a_in
        an2, t0, ss = _run(mode, an, d[p + "dn1"], d[p + "t77"], d[p + "pn"], d[p + "wn"], float(d[p + "dt"]))
        an2 = np.asarray(an2, np.float64)
        for j, il in enumerate(out_sp):
            worst[il] = max(worst.get(il, 0.0), _relerr(an2[il], d[p + "an_out"][j]))
        worst["t0"] = max(worst.get("t0", 0.0), _relerr(t0, d[p + "t0"]))
        worst["ssat"] = max(worst.get("ssat", 0.0), _ssat_err(ss, d[p + "ssat"]))
    return worst


def test_nucond_stage_parity_fp64():
    worst = _stage_errors("fp64")
    bad = {k: v for k, v in worst.items() if v > TOL64}
    assert not bad, f"fp64 NUCOND parity violations: {bad}"


def test_nucond_stage_band_fp32():
    worst = _stage_errors("fp32")
    bad = {k: v for k, v in worst.items() if v > TOL32["ssat" if k == "ssat" else "state"]}
    assert not bad, f"fp32 NUCOND band violations: {bad}"


def test_nucond_adversarial_parity_fp64():
    worst = _adv_errors("fp64")
    bad = {k: v for k, v in worst.items() if v > TOL64}
    assert not bad, f"fp64 adversarial NUCOND parity violations: {bad}"


def test_nucond_adversarial_band_fp32():
    worst = _adv_errors("fp32")
    # fp32 RK2c trajectories with exact-equality exits amplify ulp differences of exp/pow; nucleated
    # number = max(0, rho*qc/m_drop - Nc) cancels -> looser band for the number moments
    bad = {k: v for k, v in worst.items() if v > (1e-3 if k in (10, 18) else TOL32["ssat" if k == "ssat" else "state"])}
    assert not bad, f"fp32 adversarial NUCOND band violations: {bad}"


@pytest.mark.parametrize("mutant", ["no_cap_qvexcess", "no_condensation_loop"])
def test_nucond_mutants_fail(monkeypatch, mutant):
    if mutant == "no_cap_qvexcess":
        real = nucond_mod.qvexcess

        def fake(qwvp0, qv0, qcw1, pres, thetap0, theta0, pi0, fcqv1, felvcp, ss1, pk, C, R):
            out = real(qwvp0, qv0, qcw1, pres, thetap0, theta0, pi0, fcqv1, felvcp, ss1, pk, C, R)
            return jnp.where(jnp.asarray(ss1) > 50.0, jnp.zeros_like(out), out)  # drop only the 631 cap call

        monkeypatch.setattr(nucond_mod, "qvexcess", fake)
    else:
        monkeypatch.setattr(nucond_mod.lax, "while_loop", lambda cond, body, init: init)
    worst = _adv_errors("fp64")
    assert max(worst.values()) > 1e-6, f"mutant {mutant} not detected: {worst}"
