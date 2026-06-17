"""Goddard GCE (mp_physics=97, ``gsfcgcescheme``) JAX port vs WRF savepoint parity.

Gates ``gpuwrf.physics.microphysics_goddard`` against the REAL WRF
``phys/module_mp_gsfcgce.F`` Fortran scheme (gsfcgce -> fall_flux + consat_s +
saticel_s, operational ihail=0/ice2=0/itaobraun=1/new_ice_sat=2 call) via the
``proofs/v090/savepoints_goddard`` single-column savepoints -- NOT a JAX
self-compare. The prognostic state (theta + 6 moist species) and surface precip
are gated against the canonical fp32 (default-REAL) WRF oracle; the fp64
transparency oracle (same UNMODIFIED source, kind-promoted to double) confirms
the fp64 JAX port matches to ~machine precision, proving the fp32 residuals are
the reference's own single-precision roundoff.

This is a no-kernel-change single-moment scheme: it reads/writes ONLY the
existing moist substrate (qv,qc,qr,qi,qs,qg) -- no new prognostic state.

Run CPU-only: JAX_PLATFORM_NAME=cpu pytest tests/test_goddard_savepoint_parity.py
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
from gpuwrf.physics.microphysics_goddard import (  # noqa: E402
    goddard_run,
    goddard_physics_tendency,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROOFS = os.path.abspath(os.path.join(HERE, "..", "proofs", "v090"))
SAVE_FP32 = os.path.join(PROOFS, "savepoints_goddard")
SAVE_FP64 = os.path.join(PROOFS, "savepoints_goddard_fp64")

CASES = (1, 2, 3, 4, 5)

# Predeclared tolerances (must match proofs/v090/run_goddard_parity.py; frozen).
TOL = {
    "th_abs": 1.0e-2,
    "q_rel": 1.0e-2,
    "q_abs_floor": 1.0e-7,
    "precip_rel": 1.5e-2,
    "precip_abs": 5.0e-4,
    "sr_abs": 1.0e-2,
    # transparency: fp64 JAX vs fp64 oracle must match to ~machine precision.
    "th_abs_fp64": 1.0e-6,
    "q_abs_fp64": 1.0e-12,
}

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT"), ("qg", "QG_OUT")]


def _have_savepoints():
    return all(os.path.exists(os.path.join(SAVE_FP32, f"goddard_case_{c}.json"))
               and os.path.exists(os.path.join(SAVE_FP64, f"goddard_case_{c}.json"))
               for c in CASES)


pytestmark = pytest.mark.skipif(
    not _have_savepoints(),
    reason="Goddard oracle savepoints missing; run proofs/v090/oracle/build_goddard_oracle.sh",
)


def _col(d, name):
    return np.asarray(d["columns"][name], dtype=np.float64)


def _run_jax(d):
    s = d["scalars"]
    g = lambda n: _col(d, n)[None, :]  # noqa: E731
    out = goddard_run(
        jnp.asarray(g("TH_IN")), jnp.asarray(g("QV_IN")), jnp.asarray(g("QC_IN")),
        jnp.asarray(g("QR_IN")), jnp.asarray(g("QI_IN")), jnp.asarray(g("QS_IN")),
        jnp.asarray(g("QG_IN")), jnp.asarray(g("RHO")), jnp.asarray(g("PII")),
        jnp.asarray(g("P")), jnp.asarray(g("Z")), jnp.asarray(g("DZ8W")), s["DT"])
    return {k: np.asarray(v)[0] for k, v in out.items()}


@pytest.mark.parametrize("cid", CASES)
def test_prognostic_state_vs_fp32_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"goddard_case_{cid}.json")) as fh:
        d = json.load(fh)
    out = _run_jax(d)

    # potential temperature
    assert np.max(np.abs(out["th"] - _col(d, "TH_OUT"))) <= TOL["th_abs"]

    # moist species: relative-to-peak OR below absolute floor
    for leaf, oname in Q_FIELDS:
        a = out[leaf]
        b = _col(d, oname)
        scale = max(np.max(np.abs(b)), TOL["q_abs_floor"])
        mad = float(np.max(np.abs(a - b)))
        assert (mad / scale <= TOL["q_rel"]) or (mad <= TOL["q_abs_floor"]), (
            f"case {cid} {leaf}: max_abs={mad:.3e} rel={mad / scale:.3e}")


@pytest.mark.parametrize("cid", CASES)
def test_surface_precip_vs_fp32_oracle(cid):
    with open(os.path.join(SAVE_FP32, f"goddard_case_{cid}.json")) as fh:
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
def test_transparency_vs_fp64_oracle(cid):
    """fp64 JAX vs fp64 (kind-promoted) WRF oracle must match ~machine precision.

    This is the key proof the fp32 residuals above are the reference's own
    single-precision roundoff, not a port defect.
    """
    with open(os.path.join(SAVE_FP64, f"goddard_case_{cid}.json")) as fh:
        d64 = json.load(fh)
    out = _run_jax(d64)
    assert np.max(np.abs(out["th"] - _col(d64, "TH_OUT"))) <= TOL["th_abs_fp64"]
    for leaf, oname in Q_FIELDS:
        mad = float(np.max(np.abs(out[leaf] - _col(d64, oname))))
        assert mad <= TOL["q_abs_fp64"], f"case {cid} {leaf}: max_abs={mad:.3e} vs fp64"


def test_physics_tendency_adapter_contract():
    """The Goddard adapter returns a valid frozen PhysicsTendency (in-place style)."""
    with open(os.path.join(SAVE_FP32, "goddard_case_2.json")) as fh:
        d = json.load(fh)
    s = d["scalars"]
    g = lambda n: _col(d, n)[None, :]  # noqa: E731
    tend = goddard_physics_tendency(
        jnp.asarray(g("TH_IN")), jnp.asarray(g("QV_IN")), jnp.asarray(g("QC_IN")),
        jnp.asarray(g("QR_IN")), jnp.asarray(g("QI_IN")), jnp.asarray(g("QS_IN")),
        jnp.asarray(g("QG_IN")), jnp.asarray(g("PII")), jnp.asarray(g("RHO")),
        jnp.asarray(g("P")), jnp.asarray(g("Z")), jnp.asarray(g("DZ8W")), s["DT"])
    assert isinstance(tend, PhysicsTendency)
    tend.validate_keys()  # raises on unknown State/accumulator key
    # in-place scheme: theta + 6 moist species are replacements, never tendencies
    assert set(tend.state_replacements) == {"theta", "qv", "qc", "qr", "qi", "qs", "qg"}
    assert tend.state_tendencies == {} or len(tend.state_tendencies) == 0
    assert set(tend.accumulator_increments) == {"rain_acc", "snow_acc", "graupel_acc"}
    # theta replacement round-trips back to the oracle TH_OUT
    theta_new = np.asarray(tend.state_replacements["theta"])[0]
    assert np.max(np.abs(theta_new - _col(d, "TH_OUT"))) <= TOL["th_abs"]


def test_registry_and_dispatch_wiring():
    """mp=97 is accepted, scan-wired (fail-closed-on-missing-deps), and routes to
    the Goddard adapter -- and is NOT confused with mp=7 (the 4-ice NUWRF scheme)."""
    from gpuwrf.contracts.physics_registry import (
        ACCEPTED_MP_PHYSICS, MP_SCHEMES, MP_MOIST_MEMBERS, MP_NUMBER_MEMBERS,
        assert_registry_consistent,
    )
    from gpuwrf.coupling.scan_adapters import MP_SCAN_ADAPTERS, goddard_adapter
    from gpuwrf.coupling.physics_dispatch import scheme_entry
    from gpuwrf.io.scheme_catalog import assert_catalog_consistent

    assert 97 in ACCEPTED_MP_PHYSICS
    assert 7 not in ACCEPTED_MP_PHYSICS  # 4-ice NUWRF scheme: NOT ported
    assert MP_SCHEMES[97].wrf_package == "gsfcgcescheme"
    assert MP_SCHEMES[97].status == "implemented"
    # no-kernel-change: existing moist substrate, no new number species
    assert MP_MOIST_MEMBERS[97] == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert MP_NUMBER_MEMBERS[97] == ()
    # scan-wired to the Goddard adapter
    assert MP_SCAN_ADAPTERS[97] is goddard_adapter
    # routable + GPU-runnable through the fail-closed dispatcher
    entry = scheme_entry("microphysics", 97)
    assert entry.gpu_runnable is True
    assert entry.entrypoint == "goddard_physics_tendency"
    # the catalog/registry invariants still hold with mp=97 added
    assert_registry_consistent()
    assert_catalog_consistent()
