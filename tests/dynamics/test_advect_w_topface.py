"""WRF ``advect_w`` top-face (lid) flux contribution tests.

Source: pristine WRF v4.7.1 ``dyn_em/module_advect_em.F`` ``advect_w``
``vert_order==3`` block (``:5996-6029``).  For the open top WRF overwrites the
top w-face flux with the 2nd-order form
``vflux(kde)=0.25*(rom(kde)+rom(kde-1))*(w(kde)+w(kde-1))`` (``:6014-6015``),
includes it in the interior face-(kde-1) divergence, and adds the lid pickup
``tend(kde) += 2*rdn(ktf)*vflux(kde)`` (``:6025-6028``).  For the rigid lid the
top-face flux is zero and there is no pickup.

These tests pin both branches with a hand-built omega/w field on the project's
one-row periodic C-grid, independent of the full dycore.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.dynamics.flux_advection import (
    CoupledVelocities,
    _vertical_flux_div_w,
    advect_w_flux,
)


def _build_field(nz: int, nx: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = jnp.asarray(rng.standard_normal((nz + 1, 1, nx)), dtype=jnp.float64)
    rom = jnp.asarray(rng.standard_normal((nz + 1, 1, nx)), dtype=jnp.float64)
    # rigid surface and top omega faces are zero in the coupled-velocity build,
    # but the open-top w-face flux uses rom(kde) which CAN be nonzero in a real
    # column; keep rom general here to exercise the top branch.
    rdn = jnp.asarray(rng.uniform(0.8, 1.2, size=(nz,)), dtype=jnp.float64)
    return w, rom, rdn


def test_topface_rigid_lid_zero_top_tendency():
    """Rigid lid: top w-face tendency stays exactly zero (closed F7 path)."""

    nz, nx = 12, 16
    w, rom, rdn = _build_field(nz, nx, seed=1)
    tend = _vertical_flux_div_w(w, rom, rdn, top_lid=True)
    top = np.asarray(tend[nz, 0, :])
    assert np.allclose(top, 0.0), f"rigid-lid top tendency must be zero, got max|{np.max(np.abs(top))}|"


def test_topface_open_top_matches_wrf_formula():
    """Open top: top-face flux + lid pickup match the WRF advect_w formula."""

    nz, nx = 12, 16
    w, rom, rdn = _build_field(nz, nx, seed=2)
    tend = _vertical_flux_div_w(w, rom, rdn, top_lid=False)

    w_np = np.asarray(w[:, 0, :])
    rom_np = np.asarray(rom[:, 0, :])
    rdn_np = np.asarray(rdn)
    vel_face = 0.5 * (rom_np + np.roll(rom_np, 1, axis=0))  # face k = 0.5*(rom(k)+rom(k-1))
    # WRF top-face 2nd-order flux: vflux(kde) = vel_face(nz)*0.5*(w(nz)+w(nz-1)).
    vflux_top = vel_face[nz, :] * 0.5 * (w_np[nz, :] + w_np[nz - 1, :])
    # WRF lid pickup: tend(kde) += 2*rdn(ktf)*vflux(kde) with ktf = nz-1.
    expected_pickup = 2.0 * rdn_np[nz - 1] * vflux_top
    # The interior loop also adds -rdn(nz-1)*(vflux(nz)-vflux(nz-1)) to face nz-1.
    top_tend = np.asarray(tend[nz, 0, :])
    # Top-face tendency = pickup ONLY (interior loop k=1..nz-1 does not touch face nz).
    assert np.allclose(top_tend, expected_pickup), (
        f"open-top lid pickup mismatch: max|d|={np.max(np.abs(top_tend - expected_pickup))}"
    )
    assert np.max(np.abs(top_tend)) > 0.0, "open-top lid pickup must be nonzero for a nonzero column"


def test_topface_interior_face_nz_minus_1_uses_top_flux():
    """The interior face nz-1 divergence must include the open-top vflux(nz)."""

    nz, nx = 12, 16
    w, rom, rdn = _build_field(nz, nx, seed=3)
    tend_lid = _vertical_flux_div_w(w, rom, rdn, top_lid=True)
    tend_open = _vertical_flux_div_w(w, rom, rdn, top_lid=False)
    # Faces 1..nz-2 are identical (vflux(nz) only enters face nz-1 and the pickup).
    d_mid = np.asarray(tend_open[1 : nz - 1, 0, :] - tend_lid[1 : nz - 1, 0, :])
    assert np.allclose(d_mid, 0.0), "faces below nz-1 must be unaffected by the top-face flux"
    d_face = np.asarray(tend_open[nz - 1, 0, :] - tend_lid[nz - 1, 0, :])
    assert np.max(np.abs(d_face)) > 0.0, "face nz-1 divergence must pick up vflux(nz) for the open top"


def test_advect_w_flux_topface_flag_threaded():
    """advect_w_flux threads top_lid through to the vertical operator."""

    nz, nx = 12, 16
    w, rom, rdn = _build_field(nz, nx, seed=4)
    ru = jnp.zeros((nz, 1, nx), dtype=jnp.float64)
    rv = jnp.zeros((nz, 1, nx), dtype=jnp.float64)
    vel = CoupledVelocities(ru=ru, rv=rv, rom=rom)
    fzm = jnp.full((nz,), 0.5, dtype=jnp.float64)
    fzp = jnp.full((nz,), 0.5, dtype=jnp.float64)
    tend_lid = advect_w_flux(w, vel, rdx=1.0, rdy=1.0, rdn=rdn, fzm=fzm, fzp=fzp, top_lid=True)
    tend_open = advect_w_flux(w, vel, rdx=1.0, rdy=1.0, rdn=rdn, fzm=fzm, fzp=fzp, top_lid=False)
    assert np.allclose(np.asarray(tend_lid[nz, 0, :]), 0.0)
    assert np.max(np.abs(np.asarray(tend_open[nz, 0, :]))) > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
