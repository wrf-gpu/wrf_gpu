"""v0.14 acoustic continuation: real-case ``rhs_ph`` horizontal advection.

Pins the WRF ``advective_order<=6`` specified-boundary branch of
``rhs_ph_wrf`` (map-factored 6th-order interior, 2nd/4th-order degradation
rows, the WRF open_x*-only gap columns, lid/top semantics) against a small
independent numpy reference that mirrors module_big_step_utilities_em.F
:1768-2072 row by row, on a synthetic terrain-following field.

The Switzerland h36 root cause (proofs/v014/switzerland_acoustic_continuation
.json): horizontal phi advection and the vertical omega/gw terms cancel ~65:1
over steep terrain, so the legacy order-2/unit-map operator (11% off on the
horizontal term) made the NET ph_tend wrong by ~7.4x its own magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from gpuwrf.dynamics.core.rhs_ph import rhs_ph_wrf  # noqa: E402


def _synthetic_fields(nz: int = 8, ny: int = 18, nx: int = 20):
    rng = np.random.default_rng(20260611)
    # terrain-following phb: smooth ridge + per-level growth
    jj, ii = np.mgrid[0:ny, 0:nx]
    ht = 1500.0 * np.exp(-(((ii - nx / 2) / 4.0) ** 2 + ((jj - ny / 2) / 5.0) ** 2))
    levels = np.linspace(0.0, 1.0, nz + 1)[:, None, None]
    phb = 9.81 * (ht[None] * (1.0 - levels) + 12000.0 * levels)
    ph = 50.0 * rng.standard_normal((nz + 1, ny, nx))
    u = 12.0 + 2.0 * rng.standard_normal((nz, ny, nx + 1))
    v = -6.0 + 2.0 * rng.standard_normal((nz, ny + 1, nx))
    w = 0.5 * rng.standard_normal((nz + 1, ny, nx))
    ww = 0.2 * rng.standard_normal((nz + 1, ny, nx))
    ww[0] = 0.0
    ww[nz] = 0.0
    mut = 65000.0 + 2000.0 * rng.standard_normal((ny, nx))
    muu = 65000.0 + 2000.0 * rng.standard_normal((ny, nx + 1))
    muv = 65000.0 + 2000.0 * rng.standard_normal((ny + 1, nx))
    c1f = np.linspace(1.0, 0.2, nz + 1)
    c2f = np.linspace(0.0, 30000.0, nz + 1)
    fnm = np.full(nz, 0.5)
    fnp = np.full(nz, 0.5)
    rdnw = np.full(nz, float(nz))
    msfux = 1.0 + 0.02 * rng.standard_normal((ny, nx + 1))
    msfvy = 1.0 + 0.02 * rng.standard_normal((ny + 1, nx))
    msfty = 1.0 + 0.02 * rng.standard_normal((ny, nx))
    return dict(
        u=u, v=v, w=w, ww=ww, ph=ph, phb=phb, mut=mut, muu=muu, muv=muv,
        c1f=c1f, c2f=c2f, fnm=fnm, fnp=fnp, rdnw=rdnw,
        msfux=msfux, msfvy=msfvy, msfty=msfty,
        rdx=1.0 / 3000.0, rdy=1.0 / 3000.0, cfn=1.4, cfn1=-0.4,
    )


def _reference_horizontal(f: dict, top_lid: bool) -> np.ndarray:
    """Independent numpy mirror of the WRF order<=6 specified branch."""

    ph_tot = f["ph"] + f["phb"]
    nzp1, ny, nx = ph_tot.shape
    nz = nzp1 - 1
    out = np.zeros_like(ph_tot)
    c1f, c2f = f["c1f"], f["c2f"]
    faces = list(range(1, nz)) + ([nz] if not top_lid else [])
    for k in faces:
        if k < nz:
            vp = f["v"][k] + f["v"][k - 1]
            up = f["u"][k] + f["u"][k - 1]
            wgt = 0.25
        else:
            vp = f["cfn"] * f["v"][nz - 1] + f["cfn1"] * f["v"][nz - 2]
            up = f["cfn"] * f["u"][nz - 1] + f["cfn1"] * f["u"][nz - 2]
            wgt = 0.5
        flow_y = (c1f[k] * f["muv"] + c2f[k]) * vp * f["msfvy"]
        flow_x = (c1f[k] * f["muu"] + c2f[k]) * up * f["msfux"]
        a = ph_tot[k]
        for j in range(ny):
            for i in range(nx):
                acc = 0.0
                # y advection rows
                fy = flow_y[j + 1, i] + flow_y[j, i]
                if 3 <= j <= ny - 4:
                    acc += fy * (45.0 * (a[j + 1, i] - a[j - 1, i])
                                 - 9.0 * (a[j + 2, i] - a[j - 2, i])
                                 + (a[j + 3, i] - a[j - 3, i])) / 60.0
                elif j in (2, ny - 3):
                    acc += fy * (8.0 * (a[j + 1, i] - a[j - 1, i])
                                 - (a[j + 2, i] - a[j - 2, i])) / 12.0
                elif j in (1, ny - 2):
                    acc += (flow_y[j + 1, i] * (a[j + 1, i] - a[j, i])
                            + flow_y[j, i] * (a[j, i] - a[j - 1, i]))
                out_y = wgt * f["rdy"] / f["msfty"][j, i] * acc
                acc = 0.0
                fx = flow_x[j, i + 1] + flow_x[j, i]
                if 3 <= i <= nx - 4:
                    acc += fx * (45.0 * (a[j, i + 1] - a[j, i - 1])
                                 - 9.0 * (a[j, i + 2] - a[j, i - 2])
                                 + (a[j, i + 3] - a[j, i - 3])) / 60.0
                elif i in (1, nx - 2):
                    acc += (flow_x[j, i + 1] * (a[j, i + 1] - a[j, i])
                            + flow_x[j, i] * (a[j, i] - a[j, i - 1]))
                # NOTE: i in (2, nx-3) intentionally gets NOTHING (WRF gates the
                # 4th-order x rows on open_x* only; specified domains skip them).
                out_x = wgt * f["rdx"] / f["msfty"][j, i] * acc
                out[k, j, i] -= out_y + out_x
    return out


def _run_branch(f: dict, top_lid: bool) -> np.ndarray:
    return np.asarray(
        rhs_ph_wrf(
            u=jnp.asarray(f["u"]), v=jnp.asarray(f["v"]), ww=jnp.asarray(f["ww"]),
            ph=jnp.asarray(f["ph"]), phb=jnp.asarray(f["phb"]), w=jnp.asarray(f["w"]),
            mut=jnp.asarray(f["mut"]), muu=jnp.asarray(f["muu"]), muv=jnp.asarray(f["muv"]),
            c1f=jnp.asarray(f["c1f"]), c2f=jnp.asarray(f["c2f"]),
            fnm=jnp.asarray(f["fnm"]), fnp=jnp.asarray(f["fnp"]), rdnw=jnp.asarray(f["rdnw"]),
            rdx=f["rdx"], rdy=f["rdy"], msfty=jnp.asarray(f["msfty"]),
            non_hydrostatic=True, advective_order=5, specified=True,
            msfux=jnp.asarray(f["msfux"]), msfvy=jnp.asarray(f["msfvy"]),
            cfn=f["cfn"], cfn1=f["cfn1"], top_lid=top_lid,
        )
    )


def _vertical_terms(f: dict, top_lid: bool) -> np.ndarray:
    """term3 + gw exactly as rhs_ph_wrf computes them (shared by both paths)."""

    ph_tot = f["ph"] + f["phb"]
    nzp1 = ph_tot.shape[0]
    nz = nzp1 - 1
    out = np.zeros_like(ph_tot)
    dphi = ph_tot[1:] - ph_tot[:-1]
    wdwn = 0.5 * (f["ww"][1:] + f["ww"][:-1]) * f["rdnw"][:, None, None] * dphi
    out[1:nz] -= f["fnm"][1:nz, None, None] * wdwn[1:nz] + f["fnp"][1:nz, None, None] * wdwn[: nz - 1]
    mass_f = f["c1f"][:, None, None] * f["mut"][None] + f["c2f"][:, None, None]
    gw = mass_f * 9.81 * f["w"] / f["msfty"][None]
    out[1:nz] += gw[1:nz]
    out[nz] = 0.0 if top_lid else gw[nz]
    return out


@pytest.mark.parametrize("top_lid", [True, False])
def test_real_case_branch_matches_independent_reference(top_lid):
    f = _synthetic_fields()
    got = _run_branch(f, top_lid)
    want = _vertical_terms(f, top_lid) + _reference_horizontal(f, top_lid)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1.0e-7)


def test_specified_gap_columns_have_no_x_advection():
    """Columns ids+2/ide-3 must carry y-advection + vertical terms only."""

    f = _synthetic_fields()
    got = _run_branch(f, True)
    # zero out u entirely -> x-advection vanishes everywhere; the gap columns
    # must be IDENTICAL with and without u (they never see x-advection).
    f_no_u = dict(f)
    f_no_u["u"] = np.zeros_like(f["u"])
    got_no_u = _run_branch(f_no_u, True)
    nx = f["mut"].shape[1]
    for col in (2, nx - 3):
        np.testing.assert_allclose(got[:, :, col], got_no_u[:, :, col], rtol=0.0, atol=1.0e-9)
    # a 6th-order interior column DOES see x-advection
    assert np.abs(got[:, :, 6] - got_no_u[:, :, 6]).max() > 1.0


def test_legacy_default_path_ignores_new_arguments():
    """advective_order=2 (default) must not require or use map factors."""

    f = _synthetic_fields()
    legacy = np.asarray(
        rhs_ph_wrf(
            u=jnp.asarray(f["u"]), v=jnp.asarray(f["v"]), ww=jnp.asarray(f["ww"]),
            ph=jnp.asarray(f["ph"]), phb=jnp.asarray(f["phb"]), w=jnp.asarray(f["w"]),
            mut=jnp.asarray(f["mut"]), muu=jnp.asarray(f["muu"]), muv=jnp.asarray(f["muv"]),
            c1f=jnp.asarray(f["c1f"]), c2f=jnp.asarray(f["c2f"]),
            fnm=jnp.asarray(f["fnm"]), fnp=jnp.asarray(f["fnp"]), rdnw=jnp.asarray(f["rdnw"]),
            rdx=f["rdx"], rdy=f["rdy"], msfty=jnp.asarray(f["msfty"]),
            non_hydrostatic=True,
        )
    )
    assert np.isfinite(legacy).all()
    # legacy path is the periodic 2nd-order operator: top face stays zero
    assert np.all(legacy[-1] == 0.0)
