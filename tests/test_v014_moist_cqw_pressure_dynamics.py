"""V0.14 moist-cqw / moist pg_buoy_w focused unit tests.

Covers the WRF-faithful moist vertical PGF/buoyancy generalization in
``gpuwrf.dynamics.core.advance_w`` and its operational gating:

1. Inertness: ``pg_buoy_w_moist`` with ``cqw_calc=0`` is BIT-IDENTICAL to
   ``pg_buoy_w_dry`` + ``dry_cqw`` (so idealized/dry gates are unchanged).
2. WRF moist form: with positive moisture the solver ``cqw`` becomes
   ``cq1 = 1/(1+0.5*(qtot_k+qtot_{k-1})) < 1`` and the extra
   ``-cq2*(c1f*mub+c2f)`` water-loading term is present with the right sign.
3. ``moist_cqw_calc_face`` is the vertical face-average of total moisture.
4. The ``GPUWRF_MOIST_CQW`` operational flag defaults ON after the GPU h1-h4
   acceptance proof, while explicit false values keep a bisection escape hatch.

CPU-only; no GPU, no fixtures.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gpuwrf.dynamics.core.advance_w import (
    dry_cqw,
    moist_cqw_calc_face,
    pg_buoy_w_dry,
    pg_buoy_w_moist,
)


def _synthetic():
    rng = np.random.default_rng(7)
    nz, ny, nx = 24, 5, 6
    p = jnp.asarray(rng.standard_normal((nz, ny, nx)) * 60.0)
    mu = jnp.asarray(rng.standard_normal((ny, nx)) * 40.0)
    mub = jnp.asarray(1.0e5 + rng.standard_normal((ny, nx)) * 200.0)
    c1f = jnp.asarray(np.linspace(1.0, 0.0, nz + 1))
    c2f = jnp.asarray(np.linspace(0.0, 1.0, nz + 1))
    rdn = jnp.asarray(1.0 / np.linspace(0.02, 0.06, nz))
    rdnw = jnp.asarray(1.0 / np.linspace(0.02, 0.06, nz))
    msfty = jnp.asarray(1.0 + rng.standard_normal((ny, nx)) * 0.02)
    return nz, ny, nx, p, mu, mub, c1f, c2f, rdn, rdnw, msfty


def test_moist_reduces_to_dry_bit_identical_when_qtot_zero():
    nz, ny, nx, p, mu, mub, c1f, c2f, rdn, rdnw, msfty = _synthetic()
    rw_dry = pg_buoy_w_dry(p, mu, c1f=c1f, rdnw=rdnw, rdn=rdn, msfty=msfty)
    rw_moist, cqw_solver = pg_buoy_w_moist(
        p, mu, mub, jnp.zeros((nz + 1, ny, nx)),
        c1f=c1f, c2f=c2f, rdnw=rdnw, rdn=rdn, msfty=msfty,
    )
    # Bit-identical rw_tend and a cqw field equal to dry_cqw (interior 1, bdry 0).
    assert float(jnp.max(jnp.abs(rw_moist - rw_dry))) == 0.0
    assert float(jnp.max(jnp.abs(cqw_solver - dry_cqw(nz, ny, nx)))) == 0.0


def test_moist_cqw_calc_face_is_vertical_face_average():
    rng = np.random.default_rng(3)
    nz, ny, nx = 10, 3, 4
    qtot = jnp.asarray(np.abs(rng.standard_normal((nz, ny, nx))) * 0.01)
    face = moist_cqw_calc_face(qtot)
    assert face.shape == (nz + 1, ny, nx)
    # interior faces 1..nz-1 = 0.5*(q[k]+q[k-1]); boundaries 0.
    expect = 0.5 * (np.asarray(qtot)[1:nz] + np.asarray(qtot)[: nz - 1])
    assert np.allclose(np.asarray(face)[1:nz], expect)
    assert float(jnp.max(jnp.abs(face[0]))) == 0.0
    assert float(jnp.max(jnp.abs(face[nz]))) == 0.0


def test_moist_adds_downward_loading_and_cq1_below_one():
    nz, ny, nx, p, mu, mub, c1f, c2f, rdn, rdnw, msfty = _synthetic()
    qtot = jnp.asarray(np.full((nz, ny, nx), 0.012))  # ~12 g/kg uniform
    cqw_calc = moist_cqw_calc_face(qtot)
    rw_dry = pg_buoy_w_dry(p, mu, c1f=c1f, rdnw=rdnw, rdn=rdn, msfty=msfty)
    rw_moist, cqw_solver = pg_buoy_w_moist(
        p, mu, mub, cqw_calc, c1f=c1f, c2f=c2f, rdnw=rdnw, rdn=rdn, msfty=msfty,
    )
    interior = slice(1, nz)
    cq1 = np.asarray(cqw_solver)[interior]
    # cqw_solver = cq1 = 1/(1+cqw_calc) strictly in (0,1) for positive moisture.
    assert np.all(cq1 < 1.0) and np.all(cq1 > 0.0)
    expect_cq1 = 1.0 / (1.0 + np.asarray(cqw_calc)[interior])
    assert np.allclose(cq1, expect_cq1)
    # The loading term -cq2*(c1f*mub+c2f) is a net DOWNWARD (negative) buoyancy on
    # the lower interior faces where the base mass (c1f*mub+c2f) is largest.
    loading = np.asarray(rw_moist - rw_dry)
    assert loading[1:6].mean() < 0.0


def test_operational_moist_cqw_flag_defaults_on(monkeypatch):
    from gpuwrf.runtime import operational_mode

    monkeypatch.delenv("GPUWRF_MOIST_CQW", raising=False)
    assert operational_mode._moist_cqw_enabled() is True
    monkeypatch.setenv("GPUWRF_MOIST_CQW", "1")
    assert operational_mode._moist_cqw_enabled() is True
    monkeypatch.setenv("GPUWRF_MOIST_CQW", "0")
    assert operational_mode._moist_cqw_enabled() is False
