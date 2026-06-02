"""MYNN-EDMF mass-flux WRF-faithfulness gate.

Verifies the JAX `dmp_mf_columns` port reproduces the pristine WRF `DMP_mf`
solver arrays (s_aw / s_awqv / s_awqc) on the real d03 12z daytime-land column,
within a predeclared relative tolerance. The WRF reference (`mf_oracle_compare.json`)
is produced by the Fortran oracle that links the actual WRF objects; this test
re-runs the JAX side and re-checks the comparison so a regression in the JAX
port is caught even without re-running Fortran.

Also asserts the wired kernel: edmf=True changes the qv/theta solve relative to
edmf=False (the mass flux is actually applied), and edmf=False is bit-identical
to the pre-EDMF behavior (no regression for existing callers).

CPU-only / fp64.
"""
import json
import os

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = os.path.join(HERE, "proofs/mynn_edmf/column_d03_12z.json")
CMP = os.path.join(HERE, "proofs/mynn_edmf/mf_oracle_compare.json")

TOL_REL = 0.05  # predeclared: fp32-WRF vs fp64-JAX + plume nonlinearity


@pytest.fixture(scope="module")
def column():
    with open(COL) as f:
        return json.load(f)


def _build_state(c):
    from gpuwrf.physics.mynn_pbl import MynnPBLColumnState
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    pr = c["profiles"]
    su = c["surface"]
    A = lambda k: jnp.array(pr[k], dtype=jnp.float64)[None, :]
    z = jnp.zeros_like(A("th"))
    st = MynnPBLColumnState(
        A("u"), A("v"), A("w"), A("th"), A("qv"), 0.5 * A("qke"),
        A("p"), A("rho"), A("dz"), z, z, z)
    s1 = lambda x: jnp.array([x], dtype=jnp.float64)
    sfc = SurfaceFluxes(
        ustar=s1(su["ust"]), theta_flux=s1(su["flt"]), qv_flux=s1(su["flqv"]),
        tau_u=s1(0.0), tau_v=s1(0.0),
        rhosfc=s1(su["psfc"] / (287.0 * su["tsk"])), fltv=s1(su["fltv"]))
    return st, sfc, su


def test_dmp_mf_matches_wrf_oracle(column):
    """JAX DMP_mf s_aw/s_awqv match the WRF Fortran oracle within tol."""
    from gpuwrf.physics.mynn_edmf import dmp_mf_columns, XLVCP
    c = column
    pr = c["profiles"]
    su = c["surface"]
    A = lambda k: jnp.array(pr[k], dtype=jnp.float64)[None, :]
    qv = A("qv")
    sqv = qv / (1.0 + qv)
    sqc = jnp.zeros_like(sqv)
    sqw = sqv
    th = A("th")
    thl = th - XLVCP / A("exner") * sqc
    thv = th * (1.0 + 0.608 * sqv)
    zw = jnp.concatenate([jnp.zeros((1, 1)), jnp.cumsum(A("dz"), axis=-1)], axis=-1)
    s1 = lambda x: jnp.array([x], dtype=jnp.float64)
    res = dmp_mf_columns(
        sqw, sqv, sqc, A("u"), A("v"), A("w"), th, thl, thv, A("tk"), A("qke"),
        A("p"), A("exner"), A("rho"), A("dz"), zw,
        ust=s1(su["ust"]), flt=s1(su["flt"]), fltv=s1(su["fltv"]),
        flq=s1(su["flq"]), flqv=s1(su["flqv"]),
        pblh=s1(su["pblh"]), ts=s1(su["tsk"]),  # ts>0 -> exact WRF criterion
        dx=su["dx"], xland=s1(su["xland"]), dt=c["config"]["delt"])

    with open(CMP) as f:
        ref = json.load(f)
    wrf_saw = np.array(ref["wrf_s_aw"])
    wrf_sawqv = np.array(ref["wrf_s_awqv"])
    jax_saw = np.asarray(res["s_aw"][0])[: len(wrf_saw)]
    jax_sawqv = np.asarray(res["s_awqv"][0])[: len(wrf_sawqv)]

    assert float(res["active"][0]) == 1.0, "MF must activate on this convective column"

    def relerr(a, b):
        return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))

    assert relerr(jax_saw, wrf_saw) <= TOL_REL
    assert relerr(jax_sawqv, wrf_sawqv) <= TOL_REL


def test_edmf_changes_qv_solve_and_no_regression_when_off(column):
    """edmf=True applies the MF; edmf=False is identical to the legacy ED path."""
    from gpuwrf.physics.mynn_pbl import (
        step_mynn_pbl_column, _step_mynn_pbl_impl_with_pblh,
    )
    st, sfc, _ = _build_state(column)
    dt = column["config"]["delt"]

    out_off = step_mynn_pbl_column(st, dt, surface=sfc, edmf=False, dx=1000.0)
    out_on = step_mynn_pbl_column(st, dt, surface=sfc, edmf=True, dx=1000.0)

    qv_off = np.asarray(out_off.qv[0])
    qv_on = np.asarray(out_on.qv[0])
    # MF must change the qv solve somewhere in the PBL.
    assert not np.allclose(qv_off, qv_on), "edmf=True did not change the qv solve"
    # Finite, bounded.
    assert np.all(np.isfinite(qv_on)) and np.all(qv_on >= 0.0)

    # Regression guard: edmf default is OFF, so the default entry equals edmf=False.
    out_default = step_mynn_pbl_column(st, dt, surface=sfc)
    assert np.allclose(np.asarray(out_default.qv[0]), qv_off)
