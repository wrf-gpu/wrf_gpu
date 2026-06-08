"""WDM5 (mp_physics=14) JAX port vs WRF savepoint parity gate.

Gates gpuwrf.physics.microphysics_wdm5 against the real WRF module_mp_wdm5.F
Fortran scheme via the proofs/v013/savepoints_wdm5 savepoints (NOT a JAX
self-compare). The prognostic mass state + surface precip are gated against the
canonical fp32 WRF oracle; the double-moment NUMBER fields (Nn, Nc, Nr) are
gated against the fp32 oracle with a slightly looser relative tolerance
(cube-root lamda inversions + 1e1..1e10 dynamic range); the categorical-floor
effective-radius diagnostics are gated against the fp64 oracle (see
proofs/v013/t3_wdm5_oracle.py for the full rationale).

WDM5 reuses the operationally-wired WDM6 Nn/Nc/Nr State leaves (no new leaf): it
is a 5-class scheme (rain+snow, NO graupel) with WSM5-style ice and the WDM
double-moment warm rain, OPERATIONALLY scan-wired (MP_SCAN_ADAPTERS[14]).

Run CPU-only:
  JAX_PLATFORM_NAME=cpu pytest tests/test_wdm5_savepoint_parity.py
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
from gpuwrf.physics.microphysics_wdm5 import (  # noqa: E402
    wdm5_run,
    wdm5_physics_tendency,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROOFS = os.path.abspath(os.path.join(HERE, "..", "proofs", "v013"))
SAVE_FP32 = os.path.join(PROOFS, "savepoints_wdm5")
SAVE_FP64 = os.path.join(PROOFS, "savepoints_wdm5_fp64")

CASES = (1, 2, 3, 4, 5, 6)

# Predeclared tolerances (must match proofs/v013/t3_wdm5_oracle.py).
TOL = {
    "t_abs": 1.0e-2,
    "q_rel": 1.0e-2, "q_abs_floor": 1.0e-7,
    "n_rel": 2.0e-2, "n_abs_floor": 1.0e2,
    "precip_rel": 1.5e-2, "precip_abs": 5.0e-4,
    "re_rel": 1.0e-2, "re_abs_floor": 1.0e-7,
    "sr_abs": 1.0e-2,
}

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT")]
N_FIELDS = [("nn", "NN_OUT"), ("nc", "NC_OUT"), ("nr", "NR_OUT")]
RE_FIELDS = [("re_cloud", "RE_CLOUD"), ("re_ice", "RE_ICE"), ("re_snow", "RE_SNOW")]


def _col(d, name):
    return np.asarray(d["columns"][name], dtype=np.float64)


def _metrics(jax_arr, oracle_arr, scale_floor):
    a = np.asarray(jax_arr, dtype=np.float64)
    b = np.asarray(oracle_arr, dtype=np.float64)
    scale = max(np.max(np.abs(b)), scale_floor)
    absdiff = np.abs(a - b)
    return float(np.max(absdiff)), float(np.max(absdiff) / scale)


def _load(cid):
    with open(os.path.join(SAVE_FP32, f"wdm5_case_{cid}.json")) as fh:
        d32 = json.load(fh)
    with open(os.path.join(SAVE_FP64, f"wdm5_case_{cid}.json")) as fh:
        d64 = json.load(fh)
    return d32, d64


def _run(d):
    s = d["scalars"]
    args = [_col(d, k)[None, :] for k in
            ("T_IN", "QV_IN", "QC_IN", "QR_IN", "QI_IN", "QS_IN",
             "NN_IN", "NC_IN", "NR_IN", "DEN", "P", "DELZ")]
    out = wdm5_run(*[jnp.asarray(a) for a in args], s["DT"])
    return {k: np.asarray(v)[0] for k, v in out.items()}


@pytest.mark.parametrize("cid", CASES)
def test_temperature(cid):
    d32, _ = _load(cid)
    out = _run(d32)
    mad, _ = _metrics(out["t"], _col(d32, "T_OUT"), 1.0)
    assert mad <= TOL["t_abs"], f"case {cid} T abs_err {mad:.3e}"


@pytest.mark.parametrize("cid", CASES)
@pytest.mark.parametrize("leaf,oname", Q_FIELDS)
def test_mass_species(cid, leaf, oname):
    d32, _ = _load(cid)
    out = _run(d32)
    mad, mrd = _metrics(out[leaf], _col(d32, oname), TOL["q_abs_floor"])
    assert (mrd <= TOL["q_rel"]) or (mad <= TOL["q_abs_floor"]), \
        f"case {cid} {leaf} rel={mrd:.3e} abs={mad:.3e}"


@pytest.mark.parametrize("cid", CASES)
@pytest.mark.parametrize("leaf,oname", N_FIELDS)
def test_number_species(cid, leaf, oname):
    """Double-moment Nn (CCN), Nc (cloud), Nr (rain)."""
    d32, _ = _load(cid)
    out = _run(d32)
    mad, mrd = _metrics(out[leaf], _col(d32, oname), TOL["n_abs_floor"])
    assert (mrd <= TOL["n_rel"]) or (mad <= TOL["n_abs_floor"]), \
        f"case {cid} {leaf} rel={mrd:.3e} abs={mad:.3e}"


@pytest.mark.parametrize("cid", CASES)
@pytest.mark.parametrize("leaf,oname", RE_FIELDS)
def test_effective_radii(cid, leaf, oname):
    """Diagnostic effective radii gated vs the fp64 oracle (floor-dust safe)."""
    _, d64 = _load(cid)
    d32, _ = _load(cid)
    out = _run(d32)
    mad, mrd = _metrics(out[leaf], _col(d64, oname), TOL["re_abs_floor"])
    assert (mrd <= TOL["re_rel"]) or (mad <= TOL["re_abs_floor"]), \
        f"case {cid} {leaf} rel={mrd:.3e} abs={mad:.3e}"


@pytest.mark.parametrize("cid", CASES)
def test_surface_precip(cid):
    d32, _ = _load(cid)
    out = _run(d32)
    s = d32["scalars"]
    for leaf, sname in [("rainncv", "RAINNCV"), ("snowncv", "SNOWNCV")]:
        ov = float(s[sname]); jv = float(out[leaf])
        tol = max(TOL["precip_rel"] * abs(ov), TOL["precip_abs"])
        assert abs(jv - ov) <= tol, f"case {cid} {leaf} jax={jv:.4e} oracle={ov:.4e}"
    assert abs(float(out["sr"]) - float(s["SR"])) <= TOL["sr_abs"], \
        f"case {cid} sr mismatch"


def test_physics_tendency_contract():
    """wdm5_physics_tendency returns a valid frozen PhysicsTendency."""
    d32, _ = _load(1)
    s = d32["scalars"]
    t = _col(d32, "T_IN")[None, :]
    pii = np.asarray(d32["columns"]["PII"])[None, :]
    theta = t / pii
    cols = {k: jnp.asarray(_col(d32, kk)[None, :]) for k, kk in [
        ("qv", "QV_IN"), ("qc", "QC_IN"), ("qr", "QR_IN"), ("qi", "QI_IN"),
        ("qs", "QS_IN"), ("nn", "NN_IN"), ("nc", "NC_IN"),
        ("nr", "NR_IN"), ("den", "DEN"), ("p", "P"), ("delz", "DELZ")]}
    tend = wdm5_physics_tendency(
        jnp.asarray(theta), cols["qv"], cols["qc"], cols["qr"], cols["qi"],
        cols["qs"], cols["nn"], cols["nc"], cols["nr"],
        jnp.asarray(pii), cols["den"], cols["p"], cols["delz"], s["DT"])
    assert isinstance(tend, PhysicsTendency)
    # in-place scheme: replacements for theta + 5-class moist + number (Nc, Nr)
    for key in ("theta", "qv", "qc", "qr", "qi", "qs", "Nr", "Nc"):
        assert key in tend.state_replacements, f"missing replacement {key}"
    # WDM5 has NO graupel -> no qg replacement, no graupel_acc
    assert "qg" not in tend.state_replacements
    for key in ("rain_acc", "snow_acc"):
        assert key in tend.accumulator_increments, f"missing accumulator {key}"
    assert "graupel_acc" not in tend.accumulator_increments
    # Nn (CCN) is threaded by the adapter into State.Nn -> carried in diagnostics
    assert "Nn" in tend.diagnostics
    for key in ("re_cloud", "re_ice", "re_snow"):
        assert key in tend.diagnostics
    tend.validate_keys()
