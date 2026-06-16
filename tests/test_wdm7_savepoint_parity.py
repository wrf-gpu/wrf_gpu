"""WDM7 (mp_physics=26) JAX port vs WRF savepoint parity gate.

Gates gpuwrf.physics.microphysics_wdm7 against the real WRF module_mp_wdm7.F
Fortran scheme via the proofs/v013_wdm7 savepoints (NOT a JAX self-compare).
The prognostic mass state (incl the precipitating HAIL class qh) + surface
precip (incl hailncv) are gated against the canonical fp32 WRF oracle; the
double-moment NUMBER fields (Nn, Nc, Nr) are gated against the fp32 oracle with
a slightly looser relative tolerance plus a predeclared fp64 floor-dust
fallback (cube-root lamda inversions + 1e1..1e10 dynamic range; the classic
fp32 scheme leaves trace leftovers at fully-converted cells where the fp64
reference and the fp64 JAX port both give 0); the categorical-floor
effective-radius diagnostics are gated against the fp64 oracle (see
proofs/v013_wdm7/run_wdm7_parity.py for the full rationale).

WDM7 = WDM6 double-moment warm rain (Nc/Nr/Nn) + a SEPARATE precipitating
single-moment HAIL class (qh; there is NO hail number Nh). The hail process
terms and the 4th semi-Lagrangian fall channel are exercised by cases 2/3/4/5
(case 4 grows surface hail).

Run CPU-only:
  JAX_PLATFORM_NAME=cpu pytest tests/test_wdm7_savepoint_parity.py
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
from gpuwrf.physics.microphysics_wdm7 import (  # noqa: E402
    wdm7_run,
    wdm7_physics_tendency,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROOFS = os.path.abspath(os.path.join(HERE, "..", "proofs", "v013_wdm7"))
SAVE_FP32 = os.path.join(PROOFS, "savepoints_wdm7")
SAVE_FP64 = os.path.join(PROOFS, "savepoints_wdm7_fp64")

CASES = (1, 2, 3, 4, 5, 6)
# slmsk per case (matches the oracle build_column): 1=land, 2=water
SLMSK = {1: 2.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 1.0, 6: 2.0}

# Predeclared tolerances (must match proofs/v013_wdm7/run_wdm7_parity.py).
TOL = {
    "t_abs": 1.0e-2,
    "q_rel": 1.0e-2, "q_abs_floor": 1.0e-7,
    "n_rel": 2.0e-2, "n_abs_floor": 1.0e2,
    "precip_rel": 1.5e-2, "precip_abs": 5.0e-4,
    "re_rel": 1.0e-2, "re_abs_floor": 1.0e-7,
    "sr_abs": 1.0e-2,
}

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT"), ("qg", "QG_OUT"), ("qh", "QH_OUT")]
N_FIELDS = [("nn", "NN_OUT"), ("nc", "NC_OUT"), ("nr", "NR_OUT")]
RE_FIELDS = [("re_cloud", "RE_CLOUD"), ("re_ice", "RE_ICE"), ("re_snow", "RE_SNOW")]

_HAVE_SAVEPOINTS = os.path.isdir(SAVE_FP32) and os.path.isfile(
    os.path.join(SAVE_FP32, "wdm7_case_1.json")) and os.path.isdir(SAVE_FP64)
pytestmark = pytest.mark.skipif(
    not _HAVE_SAVEPOINTS,
    reason="WDM7 savepoints missing; run proofs/v013_wdm7/oracle/build_wdm7_oracle.sh "
           "{fp32,fp64} first")


def _col(d, name):
    return np.asarray(d["columns"][name], dtype=np.float64)


def _metrics(jax_arr, oracle_arr, scale_floor):
    a = np.asarray(jax_arr, dtype=np.float64)
    b = np.asarray(oracle_arr, dtype=np.float64)
    scale = max(np.max(np.abs(b)), scale_floor)
    absdiff = np.abs(a - b)
    return float(np.max(absdiff)), float(np.max(absdiff) / scale)


def _load(cid):
    with open(os.path.join(SAVE_FP32, f"wdm7_case_{cid}.json")) as fh:
        d32 = json.load(fh)
    with open(os.path.join(SAVE_FP64, f"wdm7_case_{cid}.json")) as fh:
        d64 = json.load(fh)
    return d32, d64


def _run(d, cid):
    s = d["scalars"]
    args = [_col(d, k)[None, :] for k in
            ("T_IN", "QV_IN", "QC_IN", "QR_IN", "QI_IN", "QS_IN", "QG_IN", "QH_IN",
             "NN_IN", "NC_IN", "NR_IN", "DEN", "P", "DELZ")]
    slmsk = np.array([SLMSK[cid]], dtype=np.float64)
    out = wdm7_run(*[jnp.asarray(a) for a in args], s["DT"], jnp.asarray(slmsk))
    return {k: np.asarray(v)[0] for k, v in out.items()}


@pytest.mark.parametrize("cid", CASES)
def test_temperature(cid):
    d32, _ = _load(cid)
    out = _run(d32, cid)
    mad, _ = _metrics(out["t"], _col(d32, "T_OUT"), 1.0)
    assert mad <= TOL["t_abs"], f"case {cid} T abs_err {mad:.3e}"


@pytest.mark.parametrize("cid", CASES)
@pytest.mark.parametrize("leaf,oname", Q_FIELDS)
def test_mass_species(cid, leaf, oname):
    """Prognostic mass incl the precipitating hail class qh."""
    d32, _ = _load(cid)
    out = _run(d32, cid)
    mad, mrd = _metrics(out[leaf], _col(d32, oname), TOL["q_abs_floor"])
    assert (mrd <= TOL["q_rel"]) or (mad <= TOL["q_abs_floor"]), \
        f"case {cid} {leaf} rel={mrd:.3e} abs={mad:.3e}"


@pytest.mark.parametrize("cid", CASES)
@pytest.mark.parametrize("leaf,oname", N_FIELDS)
def test_number_species(cid, leaf, oname):
    """Double-moment Nn (CCN), Nc (cloud), Nr (rain).

    Passes if the field matches the fp32 OR (floor-dust fallback) the fp64
    oracle within the number tolerance (see run_wdm7_parity.py).
    """
    d32, d64 = _load(cid)
    out = _run(d32, cid)
    mad32, mrd32 = _metrics(out[leaf], _col(d32, oname), TOL["n_abs_floor"])
    mad64, mrd64 = _metrics(out[leaf], _col(d64, oname), TOL["n_abs_floor"])
    ok32 = (mrd32 <= TOL["n_rel"]) or (mad32 <= TOL["n_abs_floor"])
    ok64 = (mrd64 <= TOL["n_rel"]) or (mad64 <= TOL["n_abs_floor"])
    assert ok32 or ok64, \
        f"case {cid} {leaf} fp32 rel={mrd32:.3e} abs={mad32:.3e} | fp64 rel={mrd64:.3e}"


@pytest.mark.parametrize("cid", CASES)
@pytest.mark.parametrize("leaf,oname", RE_FIELDS)
def test_effective_radii(cid, leaf, oname):
    """Diagnostic effective radii gated vs the fp64 oracle (floor-dust safe)."""
    d32, d64 = _load(cid)
    out = _run(d32, cid)
    mad, mrd = _metrics(out[leaf], _col(d64, oname), TOL["re_abs_floor"])
    assert (mrd <= TOL["re_rel"]) or (mad <= TOL["re_abs_floor"]), \
        f"case {cid} {leaf} rel={mrd:.3e} abs={mad:.3e}"


@pytest.mark.parametrize("cid", CASES)
def test_surface_precip(cid):
    """Surface rain/snow/graupel/HAIL + sr vs the fp32 oracle."""
    d32, _ = _load(cid)
    out = _run(d32, cid)
    s = d32["scalars"]
    for leaf, sname in [("rainncv", "RAINNCV"), ("snowncv", "SNOWNCV"),
                        ("graupelncv", "GRAUPELNCV"), ("hailncv", "HAILNCV")]:
        ov = float(s[sname]); jv = float(out[leaf])
        tol = max(TOL["precip_rel"] * abs(ov), TOL["precip_abs"])
        assert abs(jv - ov) <= tol, f"case {cid} {leaf} jax={jv:.4e} oracle={ov:.4e}"
    assert abs(float(out["sr"]) - float(s["SR"])) <= TOL["sr_abs"], \
        f"case {cid} sr mismatch"


def test_physics_tendency_contract():
    """wdm7_physics_tendency returns a valid frozen PhysicsTendency.

    Must carry state replacements for theta + the moist species (incl qh) + the
    number leaves (Nc, Nr), accumulator increments for rain/snow/graupel/HAIL,
    and Nn + the effective radii in diagnostics.
    """
    d32, _ = _load(1)
    s = d32["scalars"]
    t = _col(d32, "T_IN")[None, :]
    pii = np.asarray(d32["columns"]["PII"])[None, :]
    theta = t / pii
    cols = {k: jnp.asarray(_col(d32, kk)[None, :]) for k, kk in [
        ("qv", "QV_IN"), ("qc", "QC_IN"), ("qr", "QR_IN"), ("qi", "QI_IN"),
        ("qs", "QS_IN"), ("qg", "QG_IN"), ("qh", "QH_IN"), ("nn", "NN_IN"),
        ("nc", "NC_IN"), ("nr", "NR_IN"), ("den", "DEN"), ("p", "P"),
        ("delz", "DELZ")]}
    tend = wdm7_physics_tendency(
        jnp.asarray(theta), cols["qv"], cols["qc"], cols["qr"], cols["qi"],
        cols["qs"], cols["qg"], cols["qh"], cols["nn"], cols["nc"], cols["nr"],
        jnp.asarray(pii), cols["den"], cols["p"], cols["delz"], s["DT"],
        jnp.asarray([SLMSK[1]]))
    assert isinstance(tend, PhysicsTendency)
    # in-place scheme: replacements for theta + moist (incl qh) + number (Nc, Nr)
    for key in ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh", "Nr", "Nc"):
        assert key in tend.state_replacements, f"missing replacement {key}"
    # surface precip accumulators incl hail_acc
    for key in ("rain_acc", "snow_acc", "graupel_acc", "hail_acc"):
        assert key in tend.accumulator_increments, f"missing accumulator {key}"
    # Nn is the manager-owned additive CCN leaf -> carried in diagnostics (S0)
    assert "Nn" in tend.diagnostics
    for key in ("re_cloud", "re_ice", "re_snow"):
        assert key in tend.diagnostics
    # the frozen-interface key validator must accept this payload
    tend.validate_keys()
