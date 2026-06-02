"""P0-4 Kain-Fritsch-eta cumulus parity tests against the WRF Fortran oracle.

Two gates, both against the SAME independent oracle (single-column driver linked
against the unmodified WRF module_cu_kfeta.F; savepoints in proofs/p0_4/savepoints):

  1. The NumPy reference (cumulus_kf_reference) — the line-for-line transcription
     correctness anchor. fp64 vs the REAL*4 oracle to predeclared physical tol.
  2. The JAX production port (cumulus_kf.kf_eta_para) — GPU-resident, vmappable.

Run CPU-only:  JAX_PLATFORMS=cpu taskset -c 0-3 pytest tests/test_kf_cumulus_oracle.py
"""
import json
import os
import time

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(__file__)
SAVE = os.path.join(HERE, "..", "proofs", "p0_4", "savepoints")
JAX_PROOF = os.path.join(HERE, "..", "proofs", "p0_4", "jax_parity.json")
CASES = (1, 2, 3, 4)

# PREDECLARED tolerances (frozen before comparison). fp64 vs REAL*4 oracle.
TEND_REL = 2.0e-3
TEND_ABS_FLOOR = 1.0e-9
RAINCV_REL = 3.0e-3
RAINCV_ABS = 1.0e-4
TEND_FIELDS = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQRCUTEN", "RQICUTEN", "RQSCUTEN")


def _load(cid):
    with open(os.path.join(SAVE, f"kf_case_{cid}.json")) as fh:
        return json.load(fh)


def _check_field(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    scale = max(np.max(np.abs(b)), TEND_ABS_FLOOR)
    mad = float(np.max(np.abs(a - b)))
    mrd = mad / scale
    return (mrd <= TEND_REL) or (mad <= TEND_ABS_FLOOR), mad, mrd


@pytest.mark.parametrize("cid", CASES)
def test_reference_vs_oracle(cid):
    from gpuwrf.physics import cumulus_kf_reference as ref
    d = _load(cid)
    s, c = d["scalars"], d["columns"]
    a = lambda n: np.array(c[n], dtype=np.float64)  # noqa: E731
    pi = (a("P") / 1.0e5) ** (ref.R_D / ref.CP)
    out = ref.kf_eta_para_np(a("T"), a("QV"), a("P"), a("DZ"), a("RHO"), a("W0AVG"),
                             a("U"), a("V"), s["DT"], s["DX"], pi_exner=pi, cudt=0.0,
                             warm_rain=False, f_qi=True, f_qs=True, trigger=1)
    assert int(round(s["SHALL"])) == out["ISHALL"]
    assert abs(s["CUTOP"] - out["CUTOP"]) < 0.5
    assert abs(s["CUBOT"] - out["CUBOT"]) < 0.5
    for f in TEND_FIELDS:
        ok, mad, mrd = _check_field(out[f], c[f])
        assert ok, f"{f}: max_abs={mad:.3e} max_rel={mrd:.3e}"
    assert abs(out["RAINCV"] - s["RAINCV"]) <= max(RAINCV_REL * abs(s["RAINCV"]), RAINCV_ABS)


def test_jax_vs_oracle():
    import jax.numpy as jnp
    from gpuwrf.physics import cumulus_kf as J
    started = time.perf_counter()
    cases = []
    failures = []
    for cid in CASES:
        d = _load(cid)
        s, c = d["scalars"], d["columns"]
        a = lambda n: jnp.array(c[n], dtype=jnp.float64)  # noqa: E731
        out = J.kf_eta_para(a("T"), a("QV"), a("P"), a("DZ"), a("RHO"), a("W0AVG"),
                            a("U"), a("V"), s["DT"], s["DX"], s["KX"], False, True, True)
        out = {k: (np.array(v) if hasattr(v, "shape") and v.shape else float(v)) for k, v in out.items()}
        rec = {
            "case": cid,
            "categorical": {
                "ISHALL": {"oracle": int(round(s["SHALL"])), "jax": int(out["ISHALL"])},
                "CUTOP": {"oracle": float(s["CUTOP"]), "jax": float(out["CUTOP"])},
                "CUBOT": {"oracle": float(s["CUBOT"]), "jax": float(out["CUBOT"])},
            },
            "RAINCV": {"oracle": float(s["RAINCV"]), "jax": float(out["RAINCV"])},
            "fields": {},
        }

        if rec["categorical"]["ISHALL"]["oracle"] != rec["categorical"]["ISHALL"]["jax"]:
            failures.append(f"case {cid} ISHALL")
        if abs(s["CUTOP"] - float(out["CUTOP"])) >= 0.5:
            failures.append(f"case {cid} CUTOP")
        if abs(s["CUBOT"] - float(out["CUBOT"])) >= 0.5:
            failures.append(f"case {cid} CUBOT")
        for f in TEND_FIELDS:
            ok, mad, mrd = _check_field(out[f], c[f])
            rec["fields"][f] = {"max_abs": mad, "max_rel": mrd, "pass": bool(ok)}
            if not ok:
                failures.append(f"case {cid} {f}: max_abs={mad:.3e} max_rel={mrd:.3e}")
        rain_tol = max(RAINCV_REL * abs(s["RAINCV"]), RAINCV_ABS)
        rain_abs = abs(float(out["RAINCV"]) - s["RAINCV"])
        rec["RAINCV"].update({"max_abs": float(rain_abs), "tolerance": float(rain_tol),
                              "pass": bool(rain_abs <= rain_tol)})
        if rain_abs > rain_tol:
            failures.append(f"case {cid} RAINCV: max_abs={rain_abs:.3e} tol={rain_tol:.3e}")
        rec["pass"] = not any(f"case {cid} " in failure for failure in failures)
        cases.append(rec)

    proof = {
        "verdict": "PASS" if not failures else "FAIL",
        "platform": os.environ.get("JAX_PLATFORMS", ""),
        "cases": cases,
        "elapsed_seconds": time.perf_counter() - started,
        "predeclared_tolerances": {
            "tendency_max_relative": TEND_REL,
            "tendency_abs_floor": TEND_ABS_FLOOR,
            "raincv_max_relative": RAINCV_REL,
            "raincv_abs": RAINCV_ABS,
            "categorical": "exact ISHALL, CUTOP/CUBOT within 0.5",
        },
    }
    with open(JAX_PROOF, "w") as fh:
        json.dump(proof, fh, indent=2, sort_keys=True)
        fh.write("\n")
    assert not failures, "; ".join(failures)


def test_jax_matches_reference_helpers():
    """The JAX helper primitives must match the NumPy reference helpers to
    ~machine precision (they share the same lookup tables)."""
    import random
    import jax.numpy as jnp  # noqa: F401
    from gpuwrf.physics import cumulus_kf as J
    from gpuwrf.physics import cumulus_kf_reference as R
    ALIQ = R.SVP1 * 1000.0; BLIQ = R.SVP2; CLIQ = R.SVP2 * R.SVPT0; DLIQ = R.SVP3
    random.seed(0)
    me = 0.0
    for _ in range(500):
        p = random.uniform(2e4, 1e5); thes = random.uniform(300, 380)
        tu = random.uniform(225, 310); qu = random.uniform(1e-4, 2e-2)
        ql = random.uniform(0, 5e-3); qi = random.uniform(0, 5e-3)
        ra = R.tpmix2(p, thes, tu, qu, ql, qi, R.XLV1, R.XLV0)
        ja = [float(x) for x in J.tpmix2(p, thes, tu, qu, ql, qi, J.XLV1, J.XLV0)]
        me = max(me, max(abs(x - y) for x, y in zip(ra, ja)))
        e1 = R.envirtht(p, tu, qu, ALIQ, BLIQ, CLIQ, DLIQ)
        e2 = float(J.envirtht(p, tu, qu, J.ALIQ, J.BLIQ, J.CLIQ, J.DLIQ))
        me = max(me, abs(e1 - e2))
    assert me < 1.0e-9, f"helper mismatch {me:.3e}"
