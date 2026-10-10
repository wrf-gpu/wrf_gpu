"""CAM-UW (bl_pbl_physics=9) kernel parity against the v0.3.4 pristine-WRF column oracle.

Oracle: ``proofs/v034/camuw_oracle`` (unmodified WRF objects, WRF CAM_INIT replica incl. esinti,
3 carried steps per column; 13 synthetic regimes + 13 Swiss CPU-WRF columns). The WRF scheme is
r8 internally; outputs are WRF REAL. Gate: every output within 4 REAL ulp of its per-record
field scale; integer diagnostics (kpbl, turbtype, pblh) exact; every rare branch exercised.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gpuwrf.physics.bl_camuw as cu  # noqa: E402

ORACLE = Path(__file__).resolve().parents[1] / "proofs" / "v034" / "camuw_oracle"
SETS = ("camuw_oracle_v034.json", "camuw_oracle_v034_swiss.json")
OUTS = {"rublten": "RUBLTEN", "rvblten": "RVBLTEN", "rthblten": "RTHBLTEN", "rqvblten": "RQVBLTEN",
        "rqcblten": "RQCBLTEN", "rqiblten": "RQIBLTEN", "rqniblten": "RQNIBLTEN", "tke_pbl": "TKE_PBL",
        "kvm3d": "KVM", "kvh3d": "KVH", "smaw3d": "SMAW", "turbtype3d": "TURBTYPE", "pblh": "PBLH",
        "kpbl": "KPBL", "tpert": "TPERT", "qpert": "QPERT", "wpert": "WPERT", "tauresx2d": "TAURESX",
        "tauresy2d": "TAURESY"}
EXACT = ("turbtype3d", "kpbl", "pblh")
ULP_TOL = 4.0


def _load(name):
    return json.load(open(ORACLE / name))


def _inputs(rec):
    i = rec["in"]
    a = lambda k: np.asarray(i[k], np.float32)  # noqa: E731
    return dict(dt=np.float32(60.0), u_phy=a("U"), v_phy=a("V"), th_phy=a("TH"), rho=a("RHO"), qv=a("QV"),
                qc=a("QC"), qi=a("QI"), qnc=a("QNC"), qni=a("QNI"), p_phy=a("P"), p8w=a("P8W"), z=a("Z"),
                z_at_w=a("ZW"), t_phy=a("T"), cldfra=a("CLDFRA"), rthratenlw=a("RTHRATENLW"),
                exner=a("EXNER"), wsedl=a("WSEDL"), hfx=np.float32(i["HFX"]), qfx=np.float32(i["QFX"]),
                ustar=np.float32(i["UST"]), ht=np.float32(i["HT"]), kvm3d=a("KVM"), kvh3d=a("KVH"),
                tauresx2d=np.float32(i["TAURESX"]), tauresy2d=np.float32(i["TAURESY"]),
                first_step=np.bool_(rec["step"] == 1))


@pytest.mark.parametrize("oracle_set", SETS)
def test_camuw_kernel_matches_pristine_wrf_oracle(oracle_set):
    recs = _load(oracle_set)["records"]
    assert len(recs) >= 39
    batch = {k: np.stack([_inputs(r)[k] for r in recs]) for k in _inputs(recs[0])}
    out = jax.tree_util.tree_map(np.asarray, jax.jit(jax.vmap(lambda kw: cu.camuw_column(**kw)))(batch))
    worst = {}
    for j, r in enumerate(recs):
        for k, ok in OUTS.items():
            got = np.atleast_1d(out[k][j]).astype(np.float64)
            exp = np.atleast_1d(np.asarray(r["out"][ok], np.float64))
            assert np.all(np.isfinite(got)), (k, r["name"], r["step"])
            if k in EXACT:
                np.testing.assert_array_equal(got, exp, err_msg=f"{k} {r['name']}#{r['step']}")
                continue
            scale = float(np.max(np.abs(exp)))
            ulp = float(np.spacing(np.float32(scale))) if scale > 0 else 0.0
            err = float(np.max(np.abs(got - exp)))
            tol = ULP_TOL * ulp if scale > 0 else 1e-30
            worst[k] = max(worst.get(k, 0.0), err / max(ulp, 1e-45))
            assert err <= tol, f"{k} {r['name']}#{r['step']}: err {err:.3e} > {ULP_TOL} ulp of {scale:.3e}"
    # the gate must be non-vacuous: tendencies, diffusivities and residual stress are active somewhere
    nz = {k: sum(int(np.count_nonzero(np.atleast_1d(r["out"][OUTS[k]]))) for r in recs)
          for k in ("rublten", "rthblten", "rqvblten", "rqcblten", "kvh3d", "tauresx2d")}
    assert all(v > 0 for v in nz.values()), nz


def test_camuw_oracle_exercises_every_rare_branch():
    """Census (non-vmapped so conds are real): SRCL, CL merges both ways, no-root entrainment, ..."""

    total = {}
    for oracle_set in SETS:
        recs = _load(oracle_set)["records"]
        cu._CENSUS = {}
        try:
            fn = jax.jit(lambda kw: cu.camuw_column(**kw))
            for r in recs:
                jax.block_until_ready(fn(_inputs(r)))
            jax.effects_barrier()
            for k, v in cu._CENSUS.items():
                total[k] = total.get(k, 0) + v
        finally:
            cu._CENSUS = None
    for key in ("srcl_added", "srcl_surface", "zisocl_merge_up", "zisocl_merge_down", "zisocl_extend",
                "cl_noroot", "cl_qq_negative", "cl_base_merged_with_below", "cubic_three_real_roots",
                "edge_enhanced_interfaces", "cl_surface_based"):
        assert total.get(key, 0) > 0, (key, total)
    assert total["cubic_calls"] > total["cubic_three_real_roots"]  # one-real-root branch too
    assert total["cl_total"] > total["cl_qq_negative"]  # qq >= 0 branch too
    assert total["cl_total"] > total["cl_wstar3_positive"]  # wstar3 <= 0 branch too


def test_camuw_saturation_table_and_constants_match_wrf_bitwise():
    d = _load(SETS[0])
    c = d["const"]
    assert (cu.CPAIR, cu.GRAVIT, cu.RAIR, cu.ZVIR, cu.LATVAP, cu.LATICE, cu.KARMAN, cu.EPSILO, cu.RH2O,
            cu.TMELT) == (c["cpair"], c["gravit"], c["rair"], c["zvir"], c["latvap"], c["latice"],
                          c["karman"], c["epsilo"], c["rh2o"], c["tmelt"])
    tab = cu.ESTBL
    tmin, tmax = 173.16, 375.16
    for n, expected in enumerate(d["estbl"], start=1):
        td = tmin + float(n - 1)
        e = max(min(td, tmax), tmin)
        i = int(e - tmin) + 1
        ai = float(math.trunc(e - tmin))
        got = (tmin + ai - e + 1.0) * tab[i - 1] - (tmin + ai - e) * tab[i]
        assert got == expected, (n, got, expected)
