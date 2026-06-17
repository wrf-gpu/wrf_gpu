"""SBU-YLin (mp_physics=13) JAX port vs WRF savepoint parity gate + wiring.

Gates gpuwrf.physics.microphysics_sbu_ylin against the real WRF Fortran scheme
phys/module_mp_sbu_ylin.F via the proofs/v017 savepoints (NOT a JAX self-compare).
SBU-YLin = Lin/Rutledge-Hobbs single-moment 5-class with Y. Lin's
ice-Richardson-dependent snow PSD + Liu-Daum autoconversion + Bigg freezing. The
prognostic state (theta, qv, qc, qr, qi, qs), the diagnostic ice-Richardson
profile (Ri), and the surface precip accumulator are gated against the canonical
fp32 WRF oracle.

Also gates that mp_physics=13 is an OPERATIONAL (scan-wired) option: it carries
only the standard moist members (qv,qc,qr,qi,qs) -- no new prognostic State leaf
(the Ri/Ri3D field is a pure diagnostic) -- so it passes the namelist validator,
is classified IMPLEMENTED, and routes to the sbu_ylin scan adapter.

Run CPU-only: JAX_PLATFORM_NAME=cpu pytest tests/test_sbu_ylin_savepoint_parity.py
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
from gpuwrf.physics.microphysics_sbu_ylin import (  # noqa: E402
    sbu_ylin_run,
    sbu_ylin_physics_tendency,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROOFS = os.path.abspath(os.path.join(HERE, "..", "proofs", "v017"))
SAVE_FP32 = os.path.join(PROOFS, "savepoints_sbu_ylin")

CASES = (1, 2, 3, 4, 5, 6)

# Predeclared tolerances (must match proofs/v017/run_sbu_ylin_parity.py).
TOL = {
    "th_abs": 1.0e-2,
    "q_rel": 2.0e-2,
    "q_abs_floor": 5.0e-7,
    "precip_rel": 3.0e-2,
    "precip_abs": 5.0e-4,
    "ri_abs": 1.0e-2,
}

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT")]


def _have_savepoints():
    return all(os.path.exists(os.path.join(SAVE_FP32, f"sbu_ylin_case_{c}.json"))
               for c in CASES)


_savepoints = pytest.mark.skipif(
    not _have_savepoints(),
    reason="SBU-YLin oracle savepoints missing; run proofs/v017/oracle/build_sbu_ylin_oracle.sh",
)


def _col(d, name):
    return np.asarray(d["columns"][name], dtype=np.float64)


def _run_jax(d):
    s = d["scalars"]
    args = [
        _col(d, "TH_IN")[None, :], _col(d, "QV_IN")[None, :], _col(d, "QC_IN")[None, :],
        _col(d, "QR_IN")[None, :], _col(d, "QI_IN")[None, :], _col(d, "QS_IN")[None, :],
        _col(d, "RHO")[None, :], _col(d, "PII")[None, :], _col(d, "P")[None, :],
        _col(d, "Z")[None, :], _col(d, "DZ8W")[None, :],
    ]
    ht = np.array([float(s["HT"])], dtype=np.float64)
    out = sbu_ylin_run(*[jnp.asarray(a) for a in args], jnp.asarray(ht), s["DT"])
    return {k: np.asarray(v)[0] for k, v in out.items()}


@_savepoints
@pytest.mark.parametrize("cid", CASES)
def test_prognostic_state_vs_fp32_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"sbu_ylin_case_{cid}.json")) as fh:
        d = json.load(fh)
    out = _run_jax(d)

    assert np.max(np.abs(out["th"] - _col(d, "TH_OUT"))) <= TOL["th_abs"]

    for leaf, oname in Q_FIELDS:
        a = out[leaf]
        b = _col(d, oname)
        scale = max(np.max(np.abs(b)), TOL["q_abs_floor"])
        mad = float(np.max(np.abs(a - b)))
        assert (mad / scale <= TOL["q_rel"]) or (mad <= TOL["q_abs_floor"]), (
            f"case {cid} {leaf}: max_abs={mad:.3e} rel={mad/scale:.3e}")


@_savepoints
@pytest.mark.parametrize("cid", CASES)
def test_ice_richardson_diagnostic_vs_fp32_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"sbu_ylin_case_{cid}.json")) as fh:
        d = json.load(fh)
    out = _run_jax(d)
    mad = float(np.max(np.abs(out["ri3d"] - _col(d, "RI_OUT"))))
    assert mad <= TOL["ri_abs"], f"case {cid} ri3d: max_abs={mad:.3e}"


@_savepoints
@pytest.mark.parametrize("cid", CASES)
def test_surface_precip_vs_fp32_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"sbu_ylin_case_{cid}.json")) as fh:
        d = json.load(fh)
    s = d["scalars"]
    out = _run_jax(d)
    ov = float(s["RAINNCV"])
    jv = float(out["rainncv"])
    tol = max(TOL["precip_rel"] * abs(ov), TOL["precip_abs"])
    assert abs(jv - ov) <= tol, f"case {cid} rainncv: jax={jv} ora={ov}"


@_savepoints
def test_physics_tendency_adapter_shape_and_keys():
    """The SBU-YLin tendency carries only the standard moist contract (no qg/qh).

    Unlike WSM7 (fail-closed on a qh hail leaf), SBU-YLin's PhysicsTendency
    validates cleanly: it writes only theta + qv/qc/qr/qi/qs + the standard
    rain/snow accumulators, and exposes Ri only as a diagnostic. So
    validate_keys() must NOT raise -- this is the structural reason SBU-YLin is
    operationally scan-wireable without a State/dycore/IO change.
    """
    with open(os.path.join(SAVE_FP32, "sbu_ylin_case_2.json")) as fh:
        d = json.load(fh)
    s = d["scalars"]
    args = [_col(d, "TH_IN")[None, :], _col(d, "QV_IN")[None, :], _col(d, "QC_IN")[None, :],
            _col(d, "QR_IN")[None, :], _col(d, "QI_IN")[None, :], _col(d, "QS_IN")[None, :],
            _col(d, "PII")[None, :], _col(d, "RHO")[None, :], _col(d, "P")[None, :],
            _col(d, "Z")[None, :], _col(d, "DZ8W")[None, :]]
    ht = np.array([float(s["HT"])], dtype=np.float64)
    tend = sbu_ylin_physics_tendency(*[jnp.asarray(a) for a in args], jnp.asarray(ht), s["DT"])
    assert isinstance(tend, PhysicsTendency)
    assert set(tend.state_replacements) == {"theta", "qv", "qc", "qr", "qi", "qs"}
    assert set(tend.accumulator_increments) == {"rain_acc", "snow_acc"}
    assert "ri3d" in tend.diagnostics
    # standard contract -> validate_keys must pass (no fail-closed)
    tend.validate_keys()
    # theta round-trips to the oracle TH_OUT
    th_new = np.asarray(tend.state_replacements["theta"])[0]
    assert np.max(np.abs(th_new - _col(d, "TH_OUT"))) <= TOL["th_abs"]


def test_sbu_ylin_selectable_in_namelist_validator():
    """mp_physics=13 (SBU-YLin) must PASS the namelist validator (it is wired)."""
    from gpuwrf.io.namelist_check import validate_namelist

    validate_namelist({"physics": {"mp_physics": [13, 13]}})
    # defaults + other wired schemes still pass
    validate_namelist({"physics": {"mp_physics": [8, 8]}})


def test_sbu_ylin_catalog_classification_implemented():
    """The honesty catalog must classify SBU-YLin (13) IMPLEMENTED + stay consistent."""
    from gpuwrf.io.scheme_catalog import (
        SupportStatus,
        assert_catalog_consistent,
        classify_scheme,
    )

    assert_catalog_consistent()
    s = classify_scheme("mp_physics", 13)
    assert s.status is SupportStatus.IMPLEMENTED
    assert s.wrf_name is not None and "SBU" in s.wrf_name


def test_sbu_ylin_dispatch_and_scan_wired():
    """mp=13 routes to the SBU-YLin scheme and is in the operational scan table."""
    from gpuwrf.coupling.physics_dispatch import scheme_entry
    from gpuwrf.coupling.scan_adapters import MP_SCAN_ADAPTERS, sbu_ylin_adapter

    e = scheme_entry("microphysics", 13)
    assert e.gpu_runnable
    assert "sbu_ylin" in e.entrypoint
    assert MP_SCAN_ADAPTERS[13] is sbu_ylin_adapter
