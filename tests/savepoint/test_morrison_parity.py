"""Morrison 2-moment (mp_physics=10) WRF savepoint parity test.

Runs the JAX Morrison port against the gold savepoints produced by the
single-column oracle that drives the UNMODIFIED WRF module_mp_morr_two_moment.F
(proofs/v060/oracle). Skips if the savepoints have not been generated.

Primary (binding) assertion: the fp64 JAX port reproduces the fp64 oracle build
of the SAME scheme to a machine-precision band on every field across all 6
regimes -- a faithful-transcription check, not a self-compare. Mass + precip
are additionally checked against the canonical fp32 oracle within physical tol.
"""
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SAVE_FP32 = os.path.join(ROOT, "proofs", "v060", "savepoints")
SAVE_FP64 = os.path.join(ROOT, "proofs", "v060", "savepoints_fp64")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(SAVE_FP64, "morrison_case_1.json")),
    reason="Morrison oracle savepoints not generated (run proofs/v060/oracle/build_and_run.sh)",
)

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT"), ("qg", "QG_OUT")]
N_FIELDS = [("ni", "NI_OUT"), ("ns", "NS_OUT"), ("nr", "NR_OUT"), ("ng", "NG_OUT")]

# machine-precision band for the fp64 faithfulness gate
FP64_T_ABS = 1.0e-9
FP64_Q_REL = 1.0e-9
FP64_Q_FLOOR = 1.0e-12
FP64_N_REL = 1.0e-8
FP64_N_FLOOR = 1.0e-3
# physical band for the fp32 mass/precip operational gate
FP32_T_ABS = 5.0e-2
FP32_Q_REL = 6.0e-2
FP32_Q_FLOOR = 1.0e-6


@pytest.fixture(scope="module")
def jax_mod():
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from gpuwrf.physics.microphysics_morrison import morrison_run
    return jnp, morrison_run


def _run(jnp, morrison_run, d):
    s = d["scalars"]
    c = d["columns"]

    def c1(n):
        return jnp.asarray(np.asarray(c[n], dtype=np.float64)[None, :])

    out = morrison_run(c1("TH_IN"), c1("QV_IN"), c1("QC_IN"), c1("QR_IN"),
                       c1("QI_IN"), c1("QS_IN"), c1("QG_IN"), c1("NI_IN"),
                       c1("NS_IN"), c1("NR_IN"), c1("NG_IN"), c1("PII"),
                       c1("P"), c1("DZ"), c1("W"), s["DT"])
    return out


def _col(d, n):
    return np.asarray(d["columns"][n], dtype=np.float64)


@pytest.mark.parametrize("cid", [1, 2, 3, 4, 5, 6])
def test_morrison_fp64_faithful(jax_mod, cid):
    """JAX fp64 port reproduces the fp64 WRF oracle to a machine-precision band."""
    jnp, morrison_run = jax_mod
    with open(os.path.join(SAVE_FP64, f"morrison_case_{cid}.json")) as fh:
        d = json.load(fh)
    out = _run(jnp, morrison_run, d)

    th = np.asarray(out["th"])[0]
    assert np.max(np.abs(th - _col(d, "TH_OUT"))) <= FP64_T_ABS, f"theta fp64 case {cid}"

    for leaf, oname in Q_FIELDS:
        a = np.asarray(out[leaf])[0]
        b = _col(d, oname)
        scale = max(np.max(np.abs(b)), FP64_Q_FLOOR)
        mad = np.max(np.abs(a - b))
        assert (mad / scale <= FP64_Q_REL) or (mad <= FP64_Q_FLOOR), \
            f"{leaf} fp64 case {cid}: rel={mad / scale:.3e}"

    for leaf, oname in N_FIELDS:
        a = np.asarray(out[leaf])[0]
        b = _col(d, oname)
        scale = max(np.max(np.abs(b)), FP64_N_FLOOR)
        mad = np.max(np.abs(a - b))
        assert (mad / scale <= FP64_N_REL) or (mad <= FP64_N_FLOOR), \
            f"{leaf} fp64 case {cid}: rel={mad / scale:.3e}"

    # surface precip matches fp64 oracle to a tight relative band
    rainncv = float(np.asarray(out["rainncv"])[0])
    ov = float(d["scalars"]["RAINNCV"])
    assert abs(rainncv - ov) <= max(1.0e-7 * abs(ov), 1.0e-9), f"rainncv fp64 case {cid}"


@pytest.mark.parametrize("cid", [1, 2, 3, 4, 5, 6])
def test_morrison_fp32_mass_and_precip(jax_mod, cid):
    """Mass + surface precip vs the canonical fp32 WRF oracle within physical tol."""
    jnp, morrison_run = jax_mod
    with open(os.path.join(SAVE_FP32, f"morrison_case_{cid}.json")) as fh:
        d = json.load(fh)
    out = _run(jnp, morrison_run, d)

    th = np.asarray(out["th"])[0]
    assert np.max(np.abs(th - _col(d, "TH_OUT"))) <= FP32_T_ABS, f"theta fp32 case {cid}"

    for leaf, oname in Q_FIELDS:
        a = np.asarray(out[leaf])[0]
        b = _col(d, oname)
        scale = max(np.max(np.abs(b)), FP32_Q_FLOOR)
        mad = np.max(np.abs(a - b))
        assert (mad / scale <= FP32_Q_REL) or (mad <= FP32_Q_FLOOR), \
            f"{leaf} fp32 case {cid}: rel={mad / scale:.3e}"

    rainncv = float(np.asarray(out["rainncv"])[0])
    ov = float(d["scalars"]["RAINNCV"])
    assert abs(rainncv - ov) <= max(6.0e-2 * abs(ov), 5.0e-4), f"rainncv fp32 case {cid}"


def test_morrison_adapter_keys(jax_mod):
    """The PhysicsTendency adapter returns only frozen-interface State/accumulator keys."""
    jnp, _ = jax_mod
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from gpuwrf.physics.microphysics_morrison import morrison_tendency
    with open(os.path.join(SAVE_FP64, "morrison_case_2.json")) as fh:
        d = json.load(fh)
    c = d["columns"]
    s = d["scalars"]

    def c1(n):
        return jnp.asarray(np.asarray(c[n], dtype=np.float64)[None, :])

    tend = morrison_tendency(c1("TH_IN"), c1("QV_IN"), c1("QC_IN"), c1("QR_IN"),
                             c1("QI_IN"), c1("QS_IN"), c1("QG_IN"), c1("NI_IN"),
                             c1("NS_IN"), c1("NR_IN"), c1("NG_IN"), c1("PII"),
                             c1("P"), c1("DZ"), c1("W"), s["DT"])
    tend.validate_keys()  # raises if any key is outside the frozen interface
