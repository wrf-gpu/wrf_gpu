"""WSM7 (mp_physics=24) JAX port vs WRF savepoint parity gate + fail-closed.

Gates gpuwrf.physics.microphysics_wsm7 against the real WRF Fortran scheme
phys/module_mp_wsm7.F via the proofs/v013 savepoints (NOT a JAX self-compare).
WSM7 = WSM6 single-moment rain/snow/graupel + a separate precipitating HAIL
class. The prognostic state (incl. hail qh) + surface precip (incl. hail) are
gated against the canonical fp32 WRF oracle; the categorical-floor
effective-radius diagnostics are gated against the fp64 oracle (see
proofs/v013/run_wsm7_parity.py for the rationale).

Also gates the HONESTY contract: WSM7 is parity-proven but NOT operationally
selectable (it carries a qh hail leaf the operational moist-state pytree does
not hold), so mp_physics=24 must fail-closed in the namelist validator and the
catalog must classify it RECOGNIZED_FAIL_CLOSED with a reason naming the proven
port and the hail-leaf blocker.

Run CPU-only: JAX_PLATFORM_NAME=cpu pytest tests/test_wsm7_savepoint_parity.py
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
from gpuwrf.physics.microphysics_wsm7 import (  # noqa: E402
    wsm7_run,
    wsm7_physics_tendency,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROOFS = os.path.abspath(os.path.join(HERE, "..", "proofs", "v013"))
SAVE_FP32 = os.path.join(PROOFS, "savepoints_wsm7")
SAVE_FP64 = os.path.join(PROOFS, "savepoints_wsm7_fp64")

CASES = (1, 2, 3, 4, 5, 6)

# Predeclared tolerances (must match proofs/v013/run_wsm7_parity.py).
TOL = {
    "t_abs": 5.0e-3,
    "q_rel": 6.0e-3,
    "q_abs_floor": 1.0e-7,
    "precip_rel": 1.0e-2,
    "precip_abs": 3.0e-4,
    "re_rel": 5.0e-3,
    "re_abs_floor": 1.0e-7,
    "sr_abs": 8.0e-3,
}

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT"), ("qg", "QG_OUT"), ("qh", "QH_OUT")]
RE_FIELDS = [("re_cloud", "RE_CLOUD"), ("re_ice", "RE_ICE"), ("re_snow", "RE_SNOW")]


def _have_savepoints():
    return all(os.path.exists(os.path.join(SAVE_FP32, f"wsm7_case_{c}.json"))
               and os.path.exists(os.path.join(SAVE_FP64, f"wsm7_case_{c}.json"))
               for c in CASES)


_savepoints = pytest.mark.skipif(
    not _have_savepoints(),
    reason="WSM7 oracle savepoints missing; run proofs/v013/oracle/build_wsm7_oracle.sh",
)


def _col(d, name):
    return np.asarray(d["columns"][name], dtype=np.float64)


def _run_jax(d):
    s = d["scalars"]
    args = [
        _col(d, "T_IN")[None, :], _col(d, "QV_IN")[None, :], _col(d, "QC_IN")[None, :],
        _col(d, "QR_IN")[None, :], _col(d, "QI_IN")[None, :], _col(d, "QS_IN")[None, :],
        _col(d, "QG_IN")[None, :], _col(d, "QH_IN")[None, :], _col(d, "DEN")[None, :],
        _col(d, "P")[None, :], _col(d, "DELZ")[None, :],
    ]
    out = wsm7_run(*[jnp.asarray(a) for a in args], s["DT"])
    return {k: np.asarray(v)[0] for k, v in out.items()}


@_savepoints
@pytest.mark.parametrize("cid", CASES)
def test_prognostic_state_vs_fp32_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"wsm7_case_{cid}.json")) as fh:
        d = json.load(fh)
    out = _run_jax(d)

    assert np.max(np.abs(out["t"] - _col(d, "T_OUT"))) <= TOL["t_abs"]

    for leaf, oname in Q_FIELDS:
        a = out[leaf]
        b = _col(d, oname)
        scale = max(np.max(np.abs(b)), TOL["q_abs_floor"])
        mad = float(np.max(np.abs(a - b)))
        assert (mad / scale <= TOL["q_rel"]) or (mad <= TOL["q_abs_floor"]), (
            f"case {cid} {leaf}: max_abs={mad:.3e} rel={mad/scale:.3e}")


@_savepoints
@pytest.mark.parametrize("cid", CASES)
def test_surface_precip_vs_fp32_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"wsm7_case_{cid}.json")) as fh:
        d = json.load(fh)
    s = d["scalars"]
    out = _run_jax(d)
    for leaf, sname in [("rainncv", "RAINNCV"), ("snowncv", "SNOWNCV"),
                        ("graupelncv", "GRAUPELNCV"), ("hailncv", "HAILNCV")]:
        ov = float(s[sname])
        jv = float(out[leaf])
        tol = max(TOL["precip_rel"] * abs(ov), TOL["precip_abs"])
        assert abs(jv - ov) <= tol, f"case {cid} {leaf}: jax={jv} ora={ov}"
    assert abs(float(out["sr"]) - float(s["SR"])) <= TOL["sr_abs"]


@_savepoints
@pytest.mark.parametrize("cid", CASES)
def test_effective_radii_vs_fp64_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"wsm7_case_{cid}.json")) as fh:
        d32 = json.load(fh)
    with open(os.path.join(SAVE_FP64, f"wsm7_case_{cid}.json")) as fh:
        d64 = json.load(fh)
    out = _run_jax(d32)
    for leaf, oname in RE_FIELDS:
        a = out[leaf]
        b = _col(d64, oname)
        scale = max(np.max(np.abs(b)), TOL["re_abs_floor"])
        mad = float(np.max(np.abs(a - b)))
        assert (mad / scale <= TOL["re_rel"]) or (mad <= TOL["re_abs_floor"]), (
            f"case {cid} {leaf}: max_abs={mad:.3e} rel={mad/scale:.3e}")


@_savepoints
def test_physics_tendency_adapter_shape_and_keys():
    """The WSM7 adapter returns a frozen PhysicsTendency carrying the hail leaf.

    v0.17 wired the qh hail State substrate (ADR-032) AND the hail_acc surface
    accumulator, so the WSM7 tendency's qh state-replacement and hail_acc
    accumulator increment are now BOTH valid operational contract keys:
    validate_keys() must PASS (it raised pre-v0.17).
    """
    with open(os.path.join(SAVE_FP32, "wsm7_case_4.json")) as fh:
        d = json.load(fh)
    s = d["scalars"]
    pii = _col(d, "PII")[None, :]
    theta = _col(d, "T_IN")[None, :] / pii
    args = [theta, _col(d, "QV_IN")[None, :], _col(d, "QC_IN")[None, :],
            _col(d, "QR_IN")[None, :], _col(d, "QI_IN")[None, :], _col(d, "QS_IN")[None, :],
            _col(d, "QG_IN")[None, :], _col(d, "QH_IN")[None, :], pii,
            _col(d, "DEN")[None, :], _col(d, "P")[None, :], _col(d, "DELZ")[None, :]]
    tend = wsm7_physics_tendency(*[jnp.asarray(a) for a in args], s["DT"])
    assert isinstance(tend, PhysicsTendency)
    assert set(tend.state_replacements) == {"theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"}
    assert set(tend.accumulator_increments) == {"rain_acc", "snow_acc", "graupel_acc", "hail_acc"}
    # qh and hail_acc are now in the operational State/accumulator contract.
    tend.validate_keys()
    # theta round-trips to the oracle T_OUT via pii
    theta_new = np.asarray(tend.state_replacements["theta"])[0]
    t_new = theta_new * pii[0]
    assert np.max(np.abs(t_new - _col(d, "T_OUT"))) <= TOL["t_abs"]


def test_wsm7_accepted_in_namelist_validator():
    """v0.17 wires WSM7: mp_physics=24 must now PASS the namelist validator."""
    from gpuwrf.io.namelist_check import validate_namelist

    # WSM7 is now an accepted, scan-wired scheme (was UnsupportedSchemeError).
    validate_namelist({"physics": {"mp_physics": [24, 24]}})
    # defaults + already-wired schemes still pass
    validate_namelist({"physics": {"mp_physics": [8, 8]}})
    validate_namelist({"physics": {"mp_physics": [16, 16]}})


def test_wsm7_catalog_classification_implemented():
    """The public honesty catalog must classify WSM7 (24) IMPLEMENTED in v0.17.

    The qh hail substrate (ADR-032) + hail_acc + the wsm7_adapter scan wiring
    flip WSM7 from RECOGNIZED_FAIL_CLOSED to IMPLEMENTED. The catalog must stay
    internally consistent + default mp=8 unchanged.
    """
    from gpuwrf.io.scheme_catalog import (
        SupportStatus,
        assert_catalog_consistent,
        classify_scheme,
    )

    assert_catalog_consistent()
    assert classify_scheme("mp_physics", 24).status is SupportStatus.IMPLEMENTED
    # default Thompson unchanged
    assert classify_scheme("mp_physics", 8).status is SupportStatus.IMPLEMENTED
