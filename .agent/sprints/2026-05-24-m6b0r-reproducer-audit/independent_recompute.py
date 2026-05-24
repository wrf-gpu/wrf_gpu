#!/usr/bin/env python
"""Independent recomputation of WRF calc_coef_w for the column savepoint.

This is a strict, line-by-line port of WRF dyn_em/module_small_step_em.F:570-652
written from scratch by the M6B0-R reproducer-audit tester. It is NOT derived
from scripts/m6b0r_wrf_savepoint_extract.py.

Output:
  - Independent (a, alpha, gamma) for the center column
  - Per-field max-abs delta vs:
      (i)  M6B0-R Python reproduction (in scripts/m6b0r_wrf_savepoint_extract.py)
      (ii) JAX implementation (build_epssm_column_coefficients)
  - JSON proof object dropped at proof_independent_recomputation.json.

WRF index conventions (CRITICAL):
  Fortran kts=1, kte=nz_mass. kde = nz_mass + 1.
  Mass arrays (c1h, c2h, rdn, rdnw): Fortran 1..nz_mass  -> Python 0..nz_mass-1
  W arrays   (c1f, c2f, a, alpha, gamma): Fortran 1..kde -> Python 0..kde-1
  Fortran a(i,kk,j) with kk=2..kde -> Python a[kk-1] with index 1..kde-1=nz_mass
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from netCDF4 import Dataset

import jax
import jax.numpy as jnp

from gpuwrf.dynamics.vertical_implicit_solver import build_epssm_column_coefficients
from gpuwrf.validation.savepoint_io import read_savepoint


SOURCE_WRFOUT = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/"
    "wrfout_d02_2026-05-22_00:00:00"
)
COLUMN_SP = Path(
    "/tmp/wrf_gpu2_m6b0r/.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/"
    "savepoints/column/calc_coef_w_post_step001.h5"
)


def _center_slice(size: int, width: int) -> slice:
    if width >= size:
        return slice(0, size)
    start = max((size - width) // 2, 0)
    return slice(start, start + width)


def load_column_state():
    with Dataset(SOURCE_WRFOUT) as ds:
        ny = len(ds.dimensions["south_north"])
        nx = len(ds.dimensions["west_east"])
        ys = _center_slice(ny, 1)
        xs = _center_slice(nx, 1)
        theta = np.asarray(ds.variables["T"][0, :, ys, xs], dtype=np.float64) + 300.0
        ph = np.asarray(ds.variables["PH"][0, :, ys, xs], dtype=np.float64)
        phb = np.asarray(ds.variables["PHB"][0, :, ys, xs], dtype=np.float64)
        height = (ph + phb) / 9.80665
        dz_m = np.maximum(np.abs(np.diff(height, axis=0)), 1.0)
        mut = np.asarray(
            ds.variables["MU"][0, ys, xs] + ds.variables["MUB"][0, ys, xs],
            dtype=np.float64,
        )
        c1h = np.asarray(ds.variables["C1H"][0], dtype=np.float64)
        c2h = np.asarray(ds.variables["C2H"][0], dtype=np.float64)
        c1f = np.asarray(ds.variables["C1F"][0], dtype=np.float64)
        c2f = np.asarray(ds.variables["C2F"][0], dtype=np.float64)
        rdn = np.asarray(ds.variables["RDN"][0], dtype=np.float64)
        rdnw = np.asarray(ds.variables["RDNW"][0], dtype=np.float64)
        top_lid = bool(getattr(ds, "TOP_LID", True))  # absent attr -> True (matches extractor)
        dt = float(getattr(ds, "DT", 6.0))
        epssm = float(getattr(ds, "EPSSM", 0.1) or 0.1)
    return {
        "theta": theta,
        "dz_m": dz_m,
        "mut": mut,
        "c1h": c1h,
        "c2h": c2h,
        "c1f": c1f,
        "c2f": c2f,
        "rdn": rdn,
        "rdnw": rdnw,
        "top_lid": top_lid,
        "dt": dt,
        "epssm": epssm,
    }


def independent_calc_coef_w(state, *, g=9.80665):
    """Strict translation of WRF :570-652.

    Uses cqw = 1.0 and c2a = 1.0 placeholders (identical to M6B0-R extractor)
    so that any differences from the extractor are NOT due to cqw/c2a but to
    the calc_coef_w arithmetic itself.
    """
    mut = state["mut"]  # (ny, nx)
    c1h = state["c1h"]  # Python 0-indexed (length nz_mass)
    c2h = state["c2h"]
    c1f = state["c1f"]  # Python 0-indexed (length nz_mass+1 = kde)
    c2f = state["c2f"]
    rdn = state["rdn"]
    rdnw = state["rdnw"]
    dts = state["dt"]
    epssm = state["epssm"]
    top_lid = state["top_lid"]

    nz_mass = int(state["theta"].shape[0])  # WRF Fortran kte = nz_mass; kde = nz_mass+1
    kde = nz_mass + 1  # Fortran value
    ny, nx = mut.shape

    # W-face arrays sized 1..kde -> Python 0..kde-1 -> length kde
    a = np.zeros((kde, ny, nx), dtype=np.float64)
    alpha = np.ones((kde, ny, nx), dtype=np.float64)
    gamma = np.zeros((kde, ny, nx), dtype=np.float64)

    # Placeholders matching M6B0-R extractor
    cqw = np.ones((kde, ny, nx), dtype=np.float64)
    c2a = np.ones((nz_mass, ny, nx), dtype=np.float64)

    cof = (0.5 * dts * g * (1.0 + epssm)) ** 2

    # WRF: lid_flag=1; IF(top_lid) lid_flag=0
    lid_flag = 0.0 if top_lid else 1.0

    # Fortran a(i,2,j) = 0
    a[1, :, :] = 0.0  # Fortran index 2 -> Python index 1

    # Fortran a(i,kde,j) with k = kde-1 (set at line 622)
    # denom uses c1h(k), c2h(k), c1f(k), c2f(k) ALL at k=kde-1
    # Python indices: c1h[kde-1-1] = c1h[nz_mass-1]; c1f[kde-1-1] = c1f[nz_mass-1]
    kF_top = kde - 1  # = nz_mass
    kP_top = kF_top - 1  # = nz_mass - 1
    denom_top_a = (
        (c1h[kP_top] * mut + c2h[kP_top])
        * (c1f[kP_top] * mut + c2f[kP_top])
    )
    # rdnw(kde-1) Fortran -> rdnw[kde-2] = rdnw[nz_mass-1]; c2a(...,kde-1,...) -> c2a[nz_mass-1]
    a[kde - 1, :, :] = (
        -2.0 * cof * rdnw[nz_mass - 1] ** 2 * c2a[nz_mass - 1] * lid_flag / denom_top_a
    )

    gamma[0, :, :] = 0.0  # Fortran gamma(i,1,j)=0 -> Python gamma[0]

    # Fortran DO kk=3, kde-1 -> kk_F in 3..kde-1
    # Python: kk_F = kk_P + 1, so kk_P in 2..kde-2 = 2..nz_mass-1 -> range(2, nz_mass)
    # Inside: k_F = kk_F - 1 -> k_P = kk_P - 1
    for kk_P in range(2, nz_mass):
        k_P = kk_P - 1
        denom = (
            (c1h[k_P] * mut + c2h[k_P]) * (c1f[k_P] * mut + c2f[k_P])
        )
        # Fortran rdn(kk) -> rdn[kk_P]; rdnw(kk-1) -> rdnw[kk_P-1]; c2a(kk-1) -> c2a[kk_P-1]
        a[kk_P, :, :] = (
            -cqw[kk_P] * cof * rdn[kk_P] * rdnw[kk_P - 1] * c2a[kk_P - 1] / denom
        )

    # Fortran DO k=2, kde-1 -> k_F in 2..kde-1; Python k_P = k_F - 1 -> 1..kde-2 = 1..nz_mass-1
    for k_P in range(1, nz_mass):
        # k_F = k_P + 1
        # denom1: c1h(k)*MUT+c2h(k) and c1f(k)*MUT+c2f(k) at k=k_F -> Python [k_P]
        denom1 = (c1h[k_P] * mut + c2h[k_P]) * (c1f[k_P] * mut + c2f[k_P])
        # denom0: c1h(k-1)*MUT+c2h(k-1) and c1f(k)*MUT+c2f(k)
        denom0 = (
            (c1h[k_P - 1] * mut + c2h[k_P - 1])
            * (c1f[k_P] * mut + c2f[k_P])
        )
        # denomp (for c): c1h(k)*MUT+c2h(k) and c1f(k+1)*MUT+c2f(k+1) -> Python [k_P] and [k_P+1]
        denomp = (
            (c1h[k_P] * mut + c2h[k_P])
            * (c1f[k_P + 1] * mut + c2f[k_P + 1])
        )
        # b = 1 + cqw(k)*cof*rdn(k)*( rdnw(k)*c2a(k)/denom1 + rdnw(k-1)*c2a(k-1)/denom0 )
        b = 1.0 + cqw[k_P] * cof * rdn[k_P] * (
            rdnw[k_P] * c2a[k_P] / denom1
            + rdnw[k_P - 1] * c2a[k_P - 1] / denom0
        )
        # c = -cqw(k)*cof*rdn(k)*rdnw(k)*c2a(k)/denomp
        c_val = -cqw[k_P] * cof * rdn[k_P] * rdnw[k_P] * c2a[k_P] / denomp
        alpha[k_P, :, :] = 1.0 / (b - a[k_P] * gamma[k_P - 1])
        gamma[k_P, :, :] = c_val * alpha[k_P]

    # Fortran top: k=kde
    # b = 1 + 2*cof*rdnw(kde-1)^2*c2a(kde-1) / ((c1h(k-1)*MUT+c2h(k-1))*(c1f(k)*MUT+c2f(k)))
    # k=kde -> c1h(k-1) = c1h(kde-1) = c1h(nz_mass) -> Python c1h[nz_mass-1]
    # c1f(k) = c1f(kde) = c1f(nz_mass+1) -> Python c1f[nz_mass]
    denom_top_b = (
        (c1h[nz_mass - 1] * mut + c2h[nz_mass - 1])
        * (c1f[nz_mass] * mut + c2f[nz_mass])
    )
    b_top = (
        1.0 + 2.0 * cof * rdnw[nz_mass - 1] ** 2 * c2a[nz_mass - 1] / denom_top_b
    )
    # alpha(kde) = 1/(b - a(kde)*gamma(kde-1)) -> Python alpha[kde-1] = alpha[nz_mass]
    alpha[kde - 1, :, :] = 1.0 / (b_top - a[kde - 1] * gamma[kde - 2])
    gamma[kde - 1, :, :] = 0.0  # c=0 so gamma=0

    return {"a": a, "alpha": alpha, "gamma": gamma}


def m6b0r_reproduction(state):
    """Re-call the extractor's _wrf_calc_coef_w to get its expected coeffs."""
    # Import from extractor module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extractor", str(ROOT / "scripts" / "m6b0r_wrf_savepoint_extract.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._wrf_calc_coef_w(state, dts=state["dt"], epssm=state["epssm"])


def jax_coefs(state):
    coeffs = build_epssm_column_coefficients(
        jnp.asarray(state["theta"]),
        jnp.asarray(state["dz_m"]),
        dt=state["dt"],
        epssm=state["epssm"],
    )
    _cofrz, _cofwr, _cofwz, _coftz, _cofwt, _rdzw, tri_a, tri_b, tri_c = [
        np.asarray(jax.device_get(x)) for x in coeffs
    ]
    return {"a": tri_a, "alpha": 1.0 / tri_b, "gamma": tri_c / tri_b}


def field_delta(a, b, name):
    common = tuple(min(x, y) for x, y in zip(a.shape, b.shape))
    sl = tuple(slice(0, n) for n in common)
    d = a[sl] - b[sl]
    abs_d = np.abs(d)
    max_abs = float(np.nanmax(abs_d)) if abs_d.size else 0.0
    loc = (
        [int(x) for x in np.unravel_index(int(np.nanargmax(abs_d)), d.shape)]
        if abs_d.size
        else []
    )
    return {"max_abs_delta": max_abs, "location": loc, "shape": list(d.shape)}


def main():
    state = load_column_state()
    print(f"nz_mass={state['theta'].shape[0]}, top_lid={state['top_lid']}, "
          f"dt={state['dt']}, epssm={state['epssm']}")

    indep = independent_calc_coef_w(state)
    m6b0r = m6b0r_reproduction(state)
    jax_out = jax_coefs(state)

    # Cross-check against committed savepoint values (which are M6B0-R reproduction output)
    sp = read_savepoint(COLUMN_SP)
    sp_arrays = {k: np.asarray(sp.arrays[k]) for k in ("a", "alpha", "gamma")}

    report = {
        "column_source": str(SOURCE_WRFOUT),
        "savepoint": str(COLUMN_SP),
        "shapes": {
            "indep": {k: list(v.shape) for k, v in indep.items()},
            "m6b0r": {k: list(v.shape) for k, v in m6b0r.items()},
            "jax": {k: list(v.shape) for k, v in jax_out.items()},
            "savepoint": {k: list(v.shape) for k, v in sp_arrays.items()},
        },
        "deltas_independent_vs_m6b0r_inmem": {
            k: field_delta(indep[k], m6b0r[k], k) for k in ("a", "alpha", "gamma")
        },
        "deltas_independent_vs_savepoint": {
            k: field_delta(indep[k], sp_arrays[k], k) for k in ("a", "alpha", "gamma")
        },
        "deltas_independent_vs_jax": {
            k: field_delta(indep[k], jax_out[k], k) for k in ("a", "alpha", "gamma")
        },
        "deltas_m6b0r_vs_jax": {
            k: field_delta(m6b0r[k], jax_out[k], k) for k in ("a", "alpha", "gamma")
        },
        "deltas_m6b0r_inmem_vs_savepoint": {
            k: field_delta(m6b0r[k], sp_arrays[k], k) for k in ("a", "alpha", "gamma")
        },
        "notes": {
            "top_lid": state["top_lid"],
            "lid_flag_indep": 0.0 if state["top_lid"] else 1.0,
            "lid_flag_m6b0r_hardcoded": 1.0,
            "cqw_placeholder": "ones (both indep and m6b0r)",
            "c2a_placeholder": "ones (both indep and m6b0r)",
        },
    }

    out = Path(__file__).parent / "proof_independent_recomputation.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
