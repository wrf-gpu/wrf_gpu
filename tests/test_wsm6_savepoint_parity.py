"""WSM6 (mp_physics=6) JAX port vs WRF savepoint parity gate.

Gates gpuwrf.physics.microphysics_wsm6 against the real WRF physics_mmm WSM6
Fortran scheme via the proofs/v060 savepoints (NOT a JAX self-compare). The
prognostic state + surface precip are gated against the canonical fp32 WRF
oracle; the categorical-floor effective-radius diagnostics are gated against
the fp64 oracle (see proofs/v060/run_wsm6_parity.py for the rationale).

Run CPU-only: JAX_PLATFORM_NAME=cpu pytest tests/test_wsm6_savepoint_parity.py
"""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.contracts.physics_interfaces import PhysicsTendency  # noqa: E402
from gpuwrf.physics.microphysics_wsm6 import (  # noqa: E402
    wsm6_run,
    wsm6_physics_tendency,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROOFS = os.path.abspath(os.path.join(HERE, "..", "proofs", "v060"))
SAVE_FP32 = os.path.join(PROOFS, "savepoints")
SAVE_FP64 = os.path.join(PROOFS, "savepoints_fp64")

CASES = (1, 2, 3, 4, 5, 6)

# Predeclared tolerances (must match proofs/v060/run_wsm6_parity.py).
TOL = {
    "t_abs": 5.0e-3,
    "q_rel": 5.0e-3,
    "q_abs_floor": 1.0e-7,
    "precip_rel": 8.0e-3,
    "precip_abs": 2.0e-4,
    "re_rel": 5.0e-3,
    "re_abs_floor": 1.0e-7,
    "sr_abs": 5.0e-3,
}

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT"), ("qg", "QG_OUT")]
RE_FIELDS = [("re_cloud", "RE_CLOUD"), ("re_ice", "RE_ICE"), ("re_snow", "RE_SNOW")]


def _have_savepoints():
    return all(os.path.exists(os.path.join(SAVE_FP32, f"wsm6_case_{c}.json"))
               and os.path.exists(os.path.join(SAVE_FP64, f"wsm6_case_{c}.json"))
               for c in CASES)


pytestmark = pytest.mark.skipif(
    not _have_savepoints(),
    reason="WSM6 oracle savepoints missing; run proofs/v060/oracle/build_and_run.sh",
)


def _col(d, name):
    return np.asarray(d["columns"][name], dtype=np.float64)


def _run_jax(d):
    s = d["scalars"]
    args = [
        _col(d, "T_IN")[None, :], _col(d, "QV_IN")[None, :], _col(d, "QC_IN")[None, :],
        _col(d, "QR_IN")[None, :], _col(d, "QI_IN")[None, :], _col(d, "QS_IN")[None, :],
        _col(d, "QG_IN")[None, :], _col(d, "DEN")[None, :], _col(d, "P")[None, :],
        _col(d, "DELZ")[None, :],
    ]
    out = wsm6_run(*[jnp.asarray(a) for a in args], s["DT"])
    return {k: np.asarray(v)[0] for k, v in out.items()}


@pytest.mark.parametrize("cid", CASES)
def test_prognostic_state_vs_fp32_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"wsm6_case_{cid}.json")) as fh:
        d = json.load(fh)
    out = _run_jax(d)

    # temperature
    assert np.max(np.abs(out["t"] - _col(d, "T_OUT"))) <= TOL["t_abs"]

    # moist species
    for leaf, oname in Q_FIELDS:
        a = out[leaf]
        b = _col(d, oname)
        scale = max(np.max(np.abs(b)), TOL["q_abs_floor"])
        mad = float(np.max(np.abs(a - b)))
        assert (mad / scale <= TOL["q_rel"]) or (mad <= TOL["q_abs_floor"]), (
            f"case {cid} {leaf}: max_abs={mad:.3e} rel={mad/scale:.3e}")


@pytest.mark.parametrize("cid", CASES)
def test_surface_precip_vs_fp32_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"wsm6_case_{cid}.json")) as fh:
        d = json.load(fh)
    s = d["scalars"]
    out = _run_jax(d)
    for leaf, sname in [("rainncv", "RAINNCV"), ("snowncv", "SNOWNCV"),
                        ("graupelncv", "GRAUPELNCV")]:
        ov = float(s[sname])
        jv = float(out[leaf])
        tol = max(TOL["precip_rel"] * abs(ov), TOL["precip_abs"])
        assert abs(jv - ov) <= tol, f"case {cid} {leaf}: jax={jv} ora={ov}"
    assert abs(float(out["sr"]) - float(s["SR"])) <= TOL["sr_abs"]


@pytest.mark.parametrize("cid", CASES)
def test_effective_radii_vs_fp64_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"wsm6_case_{cid}.json")) as fh:
        d32 = json.load(fh)
    with open(os.path.join(SAVE_FP64, f"wsm6_case_{cid}.json")) as fh:
        d64 = json.load(fh)
    out = _run_jax(d32)
    for leaf, oname in RE_FIELDS:
        a = out[leaf]
        b = _col(d64, oname)
        scale = max(np.max(np.abs(b)), TOL["re_abs_floor"])
        mad = float(np.max(np.abs(a - b)))
        assert (mad / scale <= TOL["re_rel"]) or (mad <= TOL["re_abs_floor"]), (
            f"case {cid} {leaf}: max_abs={mad:.3e} rel={mad/scale:.3e}")


def test_physics_tendency_adapter_contract():
    """The WSM6 adapter returns a valid frozen PhysicsTendency (in-place style)."""
    with open(os.path.join(SAVE_FP32, "wsm6_case_2.json")) as fh:
        d = json.load(fh)
    s = d["scalars"]
    pii = _col(d, "PII")[None, :]
    theta = _col(d, "T_IN")[None, :] / pii
    args = [theta, _col(d, "QV_IN")[None, :], _col(d, "QC_IN")[None, :],
            _col(d, "QR_IN")[None, :], _col(d, "QI_IN")[None, :], _col(d, "QS_IN")[None, :],
            _col(d, "QG_IN")[None, :], pii, _col(d, "DEN")[None, :], _col(d, "P")[None, :],
            _col(d, "DELZ")[None, :]]
    tend = wsm6_physics_tendency(*[jnp.asarray(a) for a in args], s["DT"])
    assert isinstance(tend, PhysicsTendency)
    tend.validate_keys()  # raises on unknown State/accumulator key
    # in-place scheme: theta + moist species are replacements, never tendencies
    assert set(tend.state_replacements) == {"theta", "qv", "qc", "qr", "qi", "qs", "qg"}
    assert tend.state_tendencies == {} or len(tend.state_tendencies) == 0
    assert set(tend.accumulator_increments) == {"rain_acc", "snow_acc", "graupel_acc"}
    # theta round-trips back to the oracle T_OUT via pii
    theta_new = np.asarray(tend.state_replacements["theta"])[0]
    t_new = theta_new * pii[0]
    assert np.max(np.abs(t_new - _col(d, "T_OUT"))) <= TOL["t_abs"]
