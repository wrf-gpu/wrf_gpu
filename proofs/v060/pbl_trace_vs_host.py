"""v0.6.0 YSU + ACM2 PBL: traceable-rewrite vs host-NumPy reference cross-check.

The GPU-operational claim is that the ``jax.lax.scan``-traceable kernels
(``_ysu_column_traceable`` / ``_acm2_column_traceable``) are a 1:1 transcription
of the host-NumPy references (``_ysu_numpy`` / ``_acm2_numpy``) -- the references
are what the original v0.6.0 savepoint parity was proven on. This script runs BOTH
paths on the EXACT savepoint inputs (all 6 cases each) and reports the worst
absolute/relative difference across every output field + diagnostic.

If ``traceable == host-NumPy`` to machine precision, then (since
``host-NumPy == WRF`` within the predeclared savepoint tolerances) the
GPU-operational traceable kernel inherits the proven WRF parity -- with NO
clamp/mask/loosened-tol shortcut. This is a transcription-fidelity proof, not a
WRF self-compare; the WRF arbiter is the savepoint parity reports.

CPU JAX fp64, cores 0-3.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import gpuwrf  # noqa: E402,F401  enables x64 at import
from gpuwrf.physics.pbl_ysu import (  # noqa: E402
    YSUColumnState,
    _ysu_column_traceable,
    _ysu_numpy,
)
from gpuwrf.physics.pbl_acm2 import (  # noqa: E402
    ACM2ColumnState,
    CP_DEFAULT,
    EP1_DEFAULT,
    G_DEFAULT,
    R_D_DEFAULT,
    _acm2_column_traceable,
    _acm2_numpy,
)

ROOT = Path(__file__).resolve().parents[2]
SAVEPOINT_DIR = ROOT / "proofs" / "v060" / "savepoints"
CASES = (1, 2, 3, 4, 5, 6)
# Machine-precision-ish fp64 transcription tolerance. The two paths differ only
# in evaluation order (NumPy loops vs jnp masked ops / lax.scan), so agreement is
# expected at the ~1e-12 level for derived diagnostics and tighter for tendencies.
TRACE_ABS_TOL = 1.0e-10
TRACE_REL_TOL = 1.0e-9


def _col(data: dict, name: str) -> np.ndarray:
    return np.asarray(data["columns"][name], dtype=np.float64)


def _load(scheme: str, case_id: int) -> dict:
    with (SAVEPOINT_DIR / f"{scheme}_case_{case_id}.json").open() as fh:
        return json.load(fh)


def _diff(a, b) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
    denom = np.maximum(np.abs(b), 1e-30)
    max_rel = float(np.max(np.abs(a - b) / denom)) if a.size else 0.0
    return max_abs, max_rel


def _ysu_case(case_id: int) -> dict:
    data = _load("ysu", case_id)
    s = data["scalars"]
    state = YSUColumnState(
        u=_col(data, "U"), v=_col(data, "V"), temperature=_col(data, "T"),
        qv=_col(data, "QV"), pressure=_col(data, "P"),
        pressure_interface=np.asarray(data["columns"]["PDI"], dtype=np.float64),
        exner=_col(data, "PI"), dz=_col(data, "DZ"),
    )
    kw = dict(
        psfc=s["PSFC"], znt=s["ZNT"], ust=s["UST"], hfx=s["HFX"], qfx=s["QFX"],
        wspd=s["WSPD"], br=s["BR"], psim=s["PSIM"], psih=s["PSIH"], dt=s["DT"],
        xland=s["XLAND"], u10=s["U10"], v10=s["V10"], uoce=0.0, voce=0.0,
    )
    tend_h, diag_h = _ysu_numpy(state, **kw)
    out_t = _ysu_column_traceable(
        jnp.asarray(state.u), jnp.asarray(state.v), jnp.asarray(state.temperature),
        jnp.asarray(state.qv), jnp.asarray(state.pressure),
        jnp.asarray(state.pressure_interface), jnp.asarray(state.exner),
        jnp.asarray(state.dz), **{k: jnp.asarray(v) for k, v in kw.items()},
    )
    (u_t, v_t, th_t, qv_t, exch_h, exch_m, pblh, kpbl, wstar, delta) = out_t
    fields = {
        "RUBLTEN": (tend_h["u"], u_t), "RVBLTEN": (tend_h["v"], v_t),
        "RTHBLTEN": (tend_h["theta"], th_t), "RQVBLTEN": (tend_h["qv"], qv_t),
        "EXCH_H": (diag_h["exch_h"], exch_h), "EXCH_M": (diag_h["exch_m"], exch_m),
        "PBLH": (diag_h["pblh"], pblh), "WSTAR": (diag_h["wstar"], wstar),
        "DELTA": (diag_h["delta"], delta),
    }
    per_field = {}
    worst_abs = 0.0
    worst_rel = 0.0
    for name, (href, tref) in fields.items():
        ma, mr = _diff(tref, href)
        per_field[name] = {"max_abs": ma, "max_rel": mr}
        worst_abs = max(worst_abs, ma)
        worst_rel = max(worst_rel, mr)
    kpbl_match = int(diag_h["kpbl"]) == int(np.asarray(kpbl).reshape(()))
    return {
        "case": case_id, "kpbl_host": int(diag_h["kpbl"]),
        "kpbl_trace": int(np.asarray(kpbl).reshape(())), "kpbl_match": kpbl_match,
        "fields": per_field, "worst_abs": worst_abs, "worst_rel": worst_rel,
        "pass": bool(kpbl_match and (worst_abs <= TRACE_ABS_TOL or worst_rel <= TRACE_REL_TOL)),
    }


def _acm2_case(case_id: int) -> dict:
    data = _load("acm2", case_id)
    s = data["scalars"]
    nz = len(data["columns"]["U"])
    state = ACM2ColumnState(
        u=_col(data, "U"), v=_col(data, "V"), theta=_col(data, "THETA"),
        temperature=_col(data, "T"), qv=_col(data, "QV"), qc=_col(data, "QC"),
        qi=_col(data, "QI"), density=_col(data, "RR"), dz=_col(data, "DZ"),
    )
    kw = dict(
        pblh_initial=s["PBLH_INITIAL"], ust=s["UST"], hfx=s["HFX"], qfx=s["QFX"],
        wspd=s["WSPD"], mut=s["MUT"], dt=s["DT"], xtime=s["XTIME"],
    )
    tend_h, diag_h = _acm2_numpy(state, **kw)
    out_t = _acm2_column_traceable(
        jnp.asarray(state.u), jnp.asarray(state.v), jnp.asarray(state.theta),
        jnp.asarray(state.temperature), jnp.asarray(state.qv), jnp.asarray(state.qc),
        jnp.asarray(state.qi), jnp.asarray(state.density), jnp.asarray(state.dz),
        pblh_initial=jnp.asarray(s["PBLH_INITIAL"]), ust=jnp.asarray(s["UST"]),
        hfx=jnp.asarray(s["HFX"]), qfx=jnp.asarray(s["QFX"]), wspd=jnp.asarray(s["WSPD"]),
        mut=jnp.asarray(s["MUT"]), dt=jnp.asarray(s["DT"]), xtime=jnp.asarray(float(s["XTIME"])),
        g=jnp.asarray(G_DEFAULT), rd=jnp.asarray(R_D_DEFAULT),
        cpd=jnp.asarray(CP_DEFAULT), ep1=jnp.asarray(EP1_DEFAULT),
    )
    (u_t, v_t, th_t, qv_t, exch_h, exch_m, pblh, kpbl, regime, noconv, rmol, fh, fm) = out_t
    fields = {
        "RUBLTEN": (tend_h["u"], u_t), "RVBLTEN": (tend_h["v"], v_t),
        "RTHBLTEN": (tend_h["theta"], th_t), "RQVBLTEN": (tend_h["qv"], qv_t),
        "EXCH_H": (diag_h["exch_h"], exch_h), "EXCH_M": (diag_h["exch_m"], exch_m),
        "PBLH": (diag_h["pblh"], pblh), "RMOL": (diag_h["rmol"], rmol),
    }
    per_field = {}
    worst_abs = 0.0
    worst_rel = 0.0
    for name, (href, tref) in fields.items():
        ma, mr = _diff(tref, href)
        per_field[name] = {"max_abs": ma, "max_rel": mr}
        worst_abs = max(worst_abs, ma)
        worst_rel = max(worst_rel, mr)
    kpbl_match = int(diag_h["kpbl"]) == int(np.asarray(kpbl).reshape(()))
    noconv_match = int(diag_h["noconv"]) == int(np.asarray(noconv).reshape(()))
    return {
        "case": case_id, "kpbl_host": int(diag_h["kpbl"]),
        "kpbl_trace": int(np.asarray(kpbl).reshape(())), "kpbl_match": kpbl_match,
        "noconv_host": int(diag_h["noconv"]), "noconv_trace": int(np.asarray(noconv).reshape(())),
        "noconv_match": noconv_match,
        "fields": per_field, "worst_abs": worst_abs, "worst_rel": worst_rel,
        "pass": bool(kpbl_match and noconv_match
                     and (worst_abs <= TRACE_ABS_TOL or worst_rel <= TRACE_REL_TOL)),
    }


def run() -> dict:
    ysu_cases = [_ysu_case(c) for c in CASES]
    acm2_cases = [_acm2_case(c) for c in CASES]
    ysu_worst_abs = max(c["worst_abs"] for c in ysu_cases)
    acm2_worst_abs = max(c["worst_abs"] for c in acm2_cases)
    return {
        "proof": "v060-pbl-trace-vs-host",
        "objective": "traceable GPU-op kernel == host-NumPy reference (which == WRF "
                     "within the savepoint tolerances) -> traceable inherits WRF parity.",
        "method": "run _ysu_numpy/_acm2_numpy (host reference) AND "
                  "_ysu_column_traceable/_acm2_column_traceable on identical savepoint "
                  "inputs; report worst per-field abs/rel diff. No clamp/mask/loosened tol.",
        "predeclared_trace_tol": {"abs": TRACE_ABS_TOL, "rel": TRACE_REL_TOL,
                                  "kpbl": "exact", "noconv(acm2)": "exact"},
        "ysu": {"cases": ysu_cases, "worst_abs": ysu_worst_abs,
                "verdict": "PASS" if all(c["pass"] for c in ysu_cases) else "FAIL"},
        "acm2": {"cases": acm2_cases, "worst_abs": acm2_worst_abs,
                 "verdict": "PASS" if all(c["pass"] for c in acm2_cases) else "FAIL"},
        "all_pass": bool(all(c["pass"] for c in ysu_cases)
                         and all(c["pass"] for c in acm2_cases)),
    }


if __name__ == "__main__":
    report = run()
    out = ROOT / "proofs" / "v060" / "pbl_trace_vs_host.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["all_pass"] else 1)
