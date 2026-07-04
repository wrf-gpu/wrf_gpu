"""Aerosol-aware Morrison 2-moment (mp_physics=40) WRF savepoint parity test.

Runs the JAX aero-Morrison port against the gold savepoints produced by the
single-column oracle that drives the UNMODIFIED WRF
module_mp_morr_two_moment_aero.F (proofs/v023/oracle/morraero, aercu_opt=2,
IACT=4, INUC=2, prognostic NC).  Skips if the savepoints are absent.

Gates (modeled on test_morrison_parity.py):
1. fp64 faithfulness (BINDING): the fp64 JAX port reproduces the fp64 oracle
   build (-fdefault-real-8) to a machine-precision band on every field across
   all 6 regimes, including the 12th prognostic NC and the aero diagnostics
   (EFCG/EFIG/EFSG/WACT/CCN1..7).
2. fp32 physical band vs the canonical fp32 oracle.  Cases 2 and 4 are the
   documented mixed-phase threshold-flip cases: the reference scheme's OWN
   fp32-vs-fp64 pair diverges there (RAINNCV rel 4.3%/5.9%, |dQ|~1e-4 at
   single levels — see proofs/v023/oracle/morraero/README.md), so the fp64
   port lands the same distance from the fp32 gold (measured: RAINNCV rel
   4.26e-2 case 2 / 5.87e-2 case 4, max q rel 4.3e-2) — inside the same
   6e-2 physical band the base-Morrison test uses; no dual-reference
   exception is needed.
3. Aero-specific activity: droplet activation engages (NC grows beyond the
   seeded input; case 3 activates from NC_IN=0), CCN spectrum is nonzero and
   monotone in supersaturation, WACT is the KZH-derived sub-grid w, and the
   polluted case 4 activates more droplets than the clean case 6 — so the
   suite cannot pass on a base-Morrison-only path.
"""
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SAVE = os.path.join(ROOT, "proofs", "v022", "f2_oracles", "morrison_aero")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(SAVE, "morr_aero_fp64_case_1.json")),
    reason="Morrison-aero oracle savepoints not generated "
           "(run proofs/v023/oracle/morraero/morraero_build_and_run.sh)",
)

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT"),
            ("qi", "QI_OUT"), ("qs", "QS_OUT"), ("qg", "QG_OUT")]
N_FIELDS = [("ni", "NI_OUT"), ("ns", "NS_OUT"), ("nr", "NR_OUT"),
            ("ng", "NG_OUT"), ("nc", "NC_OUT")]
AERO_FIELDS = [("efcg", "EFCG_OUT"), ("efig", "EFIG_OUT"), ("efsg", "EFSG_OUT"),
               ("wact", "WACT_OUT"),
               ("ccn1", "CCN1_GS_OUT"), ("ccn2", "CCN2_GS_OUT"),
               ("ccn3", "CCN3_GS_OUT"), ("ccn4", "CCN4_GS_OUT"),
               ("ccn5", "CCN5_GS_OUT"), ("ccn6", "CCN6_GS_OUT"),
               ("ccn7", "CCN7_GS_OUT")]

# machine-precision band for the fp64 faithfulness gate (same as base port)
FP64_T_ABS = 1.0e-9
FP64_Q_REL = 1.0e-9
FP64_Q_FLOOR = 1.0e-12
FP64_N_REL = 1.0e-8
FP64_N_FLOOR = 1.0e-3
FP64_AERO_REL = 1.0e-9
FP64_AERO_FLOOR = 1.0e-3
# physical band for the fp32 mass/precip operational gate (same as base port)
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
    from gpuwrf.physics.microphysics_morrison_aero import morrison_aero_run
    return jnp, morrison_aero_run


def _load(cid, prec):
    name = (f"morr_aero_fp64_case_{cid}.json" if prec == "fp64"
            else f"morr_aero_case_{cid}.json")
    with open(os.path.join(SAVE, name)) as fh:
        return json.load(fh)


def _run(jnp, morrison_aero_run, d):
    s = d["scalars"]
    c = d["columns"]

    def c1(n):
        return jnp.asarray(np.asarray(c[n], dtype=np.float64)[None, :])

    out = morrison_aero_run(
        c1("TH_IN"), c1("QV_IN"), c1("QC_IN"), c1("QR_IN"), c1("QI_IN"),
        c1("QS_IN"), c1("QG_IN"), c1("NI_IN"), c1("NS_IN"), c1("NR_IN"),
        c1("NG_IN"), c1("NC_IN"), c1("PII"), c1("P"), c1("DZ"), c1("W"),
        c1("KZH_IN"),
        c1("AEROCU_DUST1"), c1("AEROCU_DUST2"), c1("AEROCU_DUST3"),
        c1("AEROCU_DUST4"), c1("AEROCU_SEASALT"), c1("AEROCU_SULFATE"),
        c1("AEROCU_BCPHOB"), c1("AEROCU_BCPHIL"), c1("AEROCU_OCPHOB"),
        c1("AEROCU_OCPHIL"), s["DT"], s["AERCU_FCT"])
    return out


def _col(d, n):
    return np.asarray(d["columns"][n], dtype=np.float64)


@pytest.mark.parametrize("cid", [1, 2, 3, 4, 5, 6])
def test_morrison_aero_fp64_faithful(jax_mod, cid):
    """JAX fp64 port reproduces the fp64 WRF aero oracle to machine precision."""
    jnp, run = jax_mod
    d = _load(cid, "fp64")
    out = _run(jnp, run, d)

    th = np.asarray(out["th"])[0]
    assert np.max(np.abs(th - _col(d, "TH_OUT"))) <= FP64_T_ABS, \
        f"theta fp64 case {cid}"

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

    for leaf, oname in AERO_FIELDS:
        a = np.asarray(out[leaf])[0]
        b = _col(d, oname)
        scale = max(np.max(np.abs(b)), FP64_AERO_FLOOR)
        mad = np.max(np.abs(a - b))
        assert (mad / scale <= FP64_AERO_REL) or (mad <= FP64_AERO_FLOOR), \
            f"{leaf} fp64 case {cid}: rel={mad / scale:.3e}"

    # surface precip matches the fp64 oracle to a tight relative band
    for key, oname in (("rainncv", "RAINNCV"), ("snowncv", "SNOWNCV"),
                       ("graupelncv", "GRAUPELNCV")):
        a = float(np.asarray(out[key])[0])
        ov = float(d["scalars"][oname])
        assert abs(a - ov) <= max(1.0e-7 * abs(ov), 1.0e-9), \
            f"{key} fp64 case {cid}: port={a:.9e} oracle={ov:.9e}"


@pytest.mark.parametrize("cid", [1, 2, 3, 4, 5, 6])
def test_morrison_aero_fp32_mass_and_precip(jax_mod, cid):
    """Mass + surface precip vs the canonical fp32 WRF oracle, physical band.

    Cases 2/4 carry the scheme's own fp32 single-step threshold flips (see
    module docstring); they stay inside the same 6e-2 band as the base port.
    """
    jnp, run = jax_mod
    d = _load(cid, "fp32")
    out = _run(jnp, run, d)

    th = np.asarray(out["th"])[0]
    assert np.max(np.abs(th - _col(d, "TH_OUT"))) <= FP32_T_ABS, \
        f"theta fp32 case {cid}"

    for leaf, oname in Q_FIELDS:
        a = np.asarray(out[leaf])[0]
        b = _col(d, oname)
        scale = max(np.max(np.abs(b)), FP32_Q_FLOOR)
        mad = np.max(np.abs(a - b))
        assert (mad / scale <= FP32_Q_REL) or (mad <= FP32_Q_FLOOR), \
            f"{leaf} fp32 case {cid}: rel={mad / scale:.3e}"

    rainncv = float(np.asarray(out["rainncv"])[0])
    ov = float(d["scalars"]["RAINNCV"])
    assert abs(rainncv - ov) <= max(6.0e-2 * abs(ov), 5.0e-4), \
        f"rainncv fp32 case {cid}"


def test_morrison_aero_activation_engaged(jax_mod):
    """The aerosol path is genuinely exercised — not a base-Morrison happy path.

    * Activation grows NC beyond the seeded input in every liquid-cloud case;
      case 3 starts from NC_IN identically 0 and ends with continental droplet
      numbers, which only mdm_prescribed_activate can produce.
    * The CCN diagnostic spectrum is nonzero and monotone non-decreasing with
      supersaturation (CCN1 <= ... <= CCN7).
    * WACT is the KZH-derived sub-grid vertical velocity (>= 0.1 m/s floor,
      convectively enhanced above 1 m/s in the BL) — not base-Morrison's 0.5.
    * Aerosol sensitivity: the polluted 2x-loading case 4 activates more
      droplets than the clean 0.2x case 6.
    """
    jnp, run = jax_mod
    nc_max = {}
    for cid in (1, 2, 3, 4, 6):
        d = _load(cid, "fp64")
        out = _run(jnp, run, d)
        nc_in = _col(d, "NC_IN")
        nc_out = np.asarray(out["nc"])[0]
        assert np.max(nc_out) > np.max(nc_in) * 1.01 or \
            (np.max(nc_in) == 0.0 and np.max(nc_out) > 1.0e8), \
            f"case {cid}: activation did not grow NC " \
            f"(in={np.max(nc_in):.3e}, out={np.max(nc_out):.3e})"
        nc_max[cid] = float(np.max(nc_out))

        ccn = [np.asarray(out[f"ccn{s}"])[0] for s in range(1, 8)]
        assert np.max(ccn[6]) > 0.0, f"case {cid}: CCN7 diagnostic all zero"
        for s in range(6):
            assert np.all(ccn[s + 1] >= ccn[s] - 1e-12), \
                f"case {cid}: CCN spectrum not monotone at s={s + 1}"

        wact = np.asarray(out["wact"])[0]
        assert np.min(wact) >= 0.1 - 1e-12 and np.max(wact) > 1.0, \
            f"case {cid}: WACT not a KZH-derived convective profile"

    # case 3 activates from a zero seed
    assert nc_max[3] > 1.0e8, "case 3: no activation from NC_IN=0"
    # polluted (case 4, 2x aerosol) >> clean (case 6, 0.2x aerosol)
    assert nc_max[4] > 2.0 * nc_max[6], \
        f"aerosol sensitivity missing: NC4={nc_max[4]:.3e} NC6={nc_max[6]:.3e}"
