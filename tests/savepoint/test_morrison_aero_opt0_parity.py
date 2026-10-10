"""Morrison-aerosol (mp_physics=40) at aercu_opt=0 + fp32 (WRF REAL) port gates.

aercu_opt=0 is the WRF Registry default and the only mp=40 operating point
WRF accepts without cu_physics=11 (MSKF) + the CESM aerosol climatology
(share/module_check_a_mundo.F:1407).  The wrapper then runs constant droplets
(iinum=INUM=1, NDCNST=250 cm-3), INUC=0 Cooper nucleation, no activation --
but with the aero module's DCS=350e-6 cascade and Reff defaults, so it is NOT
mp_physics=10.

Oracle: proofs/v034/f2_oracles/morrison_aero_opt0 (built by
proofs/v034/oracle/morraero_opt0/build_and_run.sh from the UNMODIFIED
module_mp_morr_two_moment_aero.F; v0.23 driver + AERCU_OPT and an added
hydrometeor-free ice-nucleation sounding CASE 7).  Cases 1-7 exist for
aercu_opt=0; case 7 for aercu_opt=0 AND 2.  The aercu_opt=2 rebuild of cases
1-6 is kept only as sha256 receipts (repro_opt2_vs_v022_{fp32,fp64}.txt) that
must equal the committed v0.23 savepoints (test below).

Gates:
1. fp64 port == fp64 oracle to the machine band (all 7 cases, both modes).
2. fp32 PORT (true REAL inputs, x64 on) vs the canonical fp32 oracle inside
   the base-port physical band, and the traced fp32 program carries zero f64
   values (exact f64 census, E169).
3. Deletion sensitivity (E39): the base mp=10 kernel and the aercu_opt=2
   kernel both FAIL the aercu_opt=0 oracle (DCS cascade / INUC / constant NC).
"""
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SAVE = os.path.join(ROOT, "proofs", "v034", "f2_oracles", "morrison_aero_opt0")
SAVE_V022 = os.path.join(ROOT, "proofs", "v022", "f2_oracles", "morrison_aero")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(SAVE, "morr_aero_opt0_fp64_case_1.json")),
    reason="aercu_opt=0 Morrison-aero oracle not generated",
)

FIELDS = ["TH", "QV", "QC", "QR", "QI", "QS", "QG", "NI", "NS", "NR", "NG", "NC"]
PRECIP = ["RAINNCV", "SNOWNCV", "GRAUPELNCV"]
IN_ORDER = ["TH_IN", "QV_IN", "QC_IN", "QR_IN", "QI_IN", "QS_IN", "QG_IN",
            "NI_IN", "NS_IN", "NR_IN", "NG_IN", "NC_IN", "PII", "P", "DZ", "W",
            "KZH_IN", "AEROCU_DUST1", "AEROCU_DUST2", "AEROCU_DUST3",
            "AEROCU_DUST4", "AEROCU_SEASALT", "AEROCU_SULFATE", "AEROCU_BCPHOB",
            "AEROCU_BCPHIL", "AEROCU_OCPHOB", "AEROCU_OCPHIL"]

# machine band (same as the base / aercu_opt=2 ports)
FP64_T_ABS = 1.0e-9
FP64_REL = 1.0e-9
FP64_PRECIP_REL = 1.0e-7
# fp32 physical band (base-port FP32 constants; mixed-phase threshold flips)
FP32_T_ABS = 5.0e-2
FP32_REL = 6.0e-2


@pytest.fixture(scope="module")
def mods():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from gpuwrf.physics.microphysics_morrison_aero import morrison_aero_run
    from gpuwrf.physics.microphysics_morrison import morrison_run
    return jax, jnp, morrison_aero_run, morrison_run


def _load(opt, prec, cid):
    tag = "fp64_" if prec == "fp64" else ""
    with open(os.path.join(SAVE, f"morr_aero_opt{opt}_{tag}case_{cid}.json")) as fh:
        return json.load(fh)


def _run(mods, d, dtype, opt):
    _, jnp, run, _ = mods
    c, s = d["columns"], d["scalars"]
    args = [jnp.asarray(np.asarray(c[n], np.float64)[None, :], dtype) for n in IN_ORDER]
    return run(*args, jnp.asarray(s["DT"], dtype), jnp.asarray(s["AERCU_FCT"], dtype),
               aercu_opt=opt)


def _errors(out, d, *, fields=FIELDS):
    """Per-field max error: |dT| for TH, max|d|/max|ref| (floored) otherwise."""
    err = {}
    for f in fields:
        a = np.asarray(out[f.lower()], np.float64)[0]
        b = np.asarray(d["columns"][f + "_OUT"], np.float64)
        if f == "TH":
            err[f] = float(np.max(np.abs(a - b)))
        else:
            floor = 1.0e-12 if f.startswith("Q") else 1.0e-3
            err[f] = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), floor))
    for k in PRECIP:
        a = float(np.asarray(out[k.lower()])[0])
        b = float(d["scalars"][k])
        err[k] = abs(a - b) / max(abs(b), 1.0e-9)
    return err


CASES = [(0, c) for c in range(1, 8)] + [(2, 7)]


@pytest.mark.parametrize("opt,cid", CASES)
def test_fp64_port_machine_band(mods, opt, cid):
    _, jnp, _, _ = mods
    d = _load(opt, "fp64", cid)
    out = _run(mods, d, jnp.float64, opt)
    err = _errors(out, d)
    assert err["TH"] <= FP64_T_ABS, err
    for f in FIELDS[1:]:
        assert err[f] <= FP64_REL, (f, err)
    for k in PRECIP:
        assert err[k] <= FP64_PRECIP_REL or abs(float(np.asarray(out[k.lower()])[0])
                                                - float(d["scalars"][k])) <= 1.0e-12, (k, err)


@pytest.mark.parametrize("opt,cid", CASES)
def test_fp32_port_physical_band(mods, opt, cid):
    _, jnp, _, _ = mods
    d = _load(opt, "fp32", cid)
    out = _run(mods, d, jnp.float32, opt)
    assert all(np.asarray(v).dtype == np.float32 for v in out.values())
    err = _errors(out, d)
    assert err["TH"] <= FP32_T_ABS, err
    for f in FIELDS[1:]:
        assert err[f] <= FP32_REL, (f, err)
    # constant droplets (opt0) / activation (opt2) are smooth: tight band
    assert err["NC"] <= 1.0e-4, err
    for k in PRECIP:
        assert err[k] <= FP32_REL or abs(float(np.asarray(out[k.lower()])[0])
                                         - float(d["scalars"][k])) <= 5.0e-4, (k, err)


@pytest.mark.parametrize("opt", [0, 2])
def test_fp32_trace_has_zero_f64(mods, opt):
    """Exact f64 census of the traced REAL program (E169): 0 f64 values."""
    jax, jnp, run, _ = mods
    c = jnp.ones((2, 4), jnp.float32)
    fn = getattr(run, "__wrapped__", run)
    cj = jax.make_jaxpr(lambda *a: fn(*a, aercu_opt=opt))(
        *([c] * 16), jnp.ones((2, 5), jnp.float32), *([c] * 10), jnp.float32(60.0))
    n64 = 0

    def walk(j):
        nonlocal n64
        for e in j.eqns:
            n64 += sum(1 for v in e.outvars
                       if getattr(v.aval, "dtype", None) == jnp.float64)
            for p in e.params.values():
                for sub in (p if isinstance(p, (list, tuple)) else [p]):
                    if hasattr(sub, "eqns"):
                        walk(sub)
                    elif hasattr(sub, "jaxpr"):
                        walk(sub.jaxpr if hasattr(sub.jaxpr, "eqns") else sub.jaxpr.jaxpr)

    walk(cj.jaxpr)
    assert n64 == 0


def test_mutant_base_mp10_fails_opt0_oracle(mods):
    """Deletion sensitivity: mp=10 (DCS=125e-6, Reff 25) != mp=40/aercu_opt=0."""
    _, jnp, _, morrison_run = mods
    worst = 0.0
    for cid in range(1, 8):
        d = _load(0, "fp64", cid)
        c = d["columns"]
        col = lambda n: jnp.asarray(np.asarray(c[n], np.float64)[None, :])  # noqa: E731
        out = morrison_run(*[col(n) for n in IN_ORDER[:11] + ["PII", "P", "DZ", "W"]],
                           jnp.asarray(d["scalars"]["DT"]))
        out = dict(out, nc=jnp.asarray(np.asarray(c["NC_OUT"])[None, :]))
        worst = max(worst, max(_errors(out, d).values()))
    assert worst > 1.0e-2, worst


def test_mutant_opt2_kernel_fails_opt0_nucleation_case(mods):
    """INUC/iinum switch is exercised: case 7 separates Cooper from Liu-Penner."""
    _, jnp, _, _ = mods
    d = _load(0, "fp64", 7)
    err = _errors(_run(mods, d, jnp.float64, 2), d)
    assert err["QI"] > 1.0 and err["NC"] > 0.5, err


@pytest.mark.parametrize("mode,tag", [("fp32", ""), ("fp64", "fp64_")])
def test_v034_build_reproduces_v022_savepoints_bytewise(mode, tag):
    """The v034 build (aercu_opt=2) reproduces the v0.23 oracle byte-for-byte."""
    import hashlib
    lines = open(os.path.join(SAVE, f"repro_opt2_vs_v022_{mode}.txt")).read().split("\n")
    rebuilt = dict(reversed(ln.split()) for ln in lines if ln.strip())
    assert len(rebuilt) == 6
    for cid in range(1, 7):
        with open(os.path.join(SAVE_V022, f"morr_aero_{tag}case_{cid}.json"), "rb") as fh:
            v022 = hashlib.sha256(fh.read()).hexdigest()
        assert rebuilt[f"morr_aero_opt2_{tag}case_{cid}.json"] == v022, cid


def test_unsupported_aercu_opt_raises(mods):
    _, jnp, run, _ = mods
    c = jnp.ones((1, 3))
    with pytest.raises(ValueError, match="aercu_opt"):
        run(*([c] * 16), jnp.ones((1, 4)), *([c] * 10), 60.0, aercu_opt=1)
