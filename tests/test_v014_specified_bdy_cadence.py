"""v0.14 SPECIFIED-domain WRF boundary cadence (stage3/wrapper sprint).

Pins the new machinery against INDEPENDENT numpy mirrors of the WRF source:

* ``specified_relax_dry_tendencies`` vs a literal per-cell re-derivation of
  ``relax_bdytend_core`` (module_bc.F:1221-1427) for the mass-coupled theta and
  the 2-D mu (corner trims, fcx/gcx taper, 5-point residual Laplacian).
* the ring-0 work-array pins reconstruct the boundary-leaf values through the
  ``small_step_finish`` algebra (the WRF spec_bdyupdate net effect).
* ``apply_lateral_boundaries(dry_spec_only=True)`` keeps only the ring-0 spec
  re-sync for dry fields, never touches p'/pb, and keeps full moisture.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State
from gpuwrf.coupling.boundary_apply import (
    BoundaryConfig,
    apply_lateral_boundaries,
    specified_relax_dry_tendencies,
    tangential_bdy_work_target_u,
    tangential_bdy_work_target_v,
)
from gpuwrf.dynamics.flux_advection import specified_flux_faces, _specified_div

CFG = BoundaryConfig()  # spec_bdy_width=5, spec_zone=1, relax_zone=4, cadence 3600


def _leaf(rng, width, z_len, side_len, scale=1.0):
    # (time=2, side=4, bdy_width, z, side_len); constant in time so the
    # interpolated strip equals the stored values at any lead.
    values = rng.normal(size=(1, 4, width, z_len, side_len)) * scale
    return jnp.asarray(np.repeat(values, 2, axis=0))


def _wrf_relax_mirror_3d(field, target, fcx, gcx, spec_zone, relax_zone):
    """Literal WRF relax_bdytend_core on a (z, ny, nx) COUPLED field.

    ``target`` is the full-ring coupled boundary target at the same grid
    locations.  Returns the tendency field (zero outside the relax rings).
    """

    z, ny, nx = field.shape
    tend = np.zeros_like(field)

    def fls(j, i, jj, ii):
        return target[:, jj, ii] - field[:, jj, ii]

    # Y-start / Y-end rows (corner ownership: i in [b_dist, nx-1-b_dist])
    for b_dist in range(spec_zone, relax_zone):
        f, g = fcx[b_dist], gcx[b_dist]
        for j, sgn in ((b_dist, +1), (ny - 1 - b_dist, -1)):
            for i in range(b_dist, nx - b_dist):
                im1, ip1 = max(i - 1, 0), min(i + 1, nx - 1)
                fls0 = target[:, j, i] - field[:, j, i]
                fls1 = target[:, j, im1] - field[:, j, im1]
                fls2 = target[:, j, ip1] - field[:, j, ip1]
                fls3 = target[:, j - sgn, i] - field[:, j - sgn, i]
                fls4 = target[:, j + sgn, i] - field[:, j + sgn, i]
                tend[:, j, i] += f * fls0 - g * (fls1 + fls2 + fls3 + fls4 - 4.0 * fls0)
    # X-start / X-end columns (j in [b_dist+1, ny-2-b_dist])
    for b_dist in range(spec_zone, relax_zone):
        f, g = fcx[b_dist], gcx[b_dist]
        for i, sgn in ((b_dist, +1), (nx - 1 - b_dist, -1)):
            for j in range(b_dist + 1, ny - 1 - b_dist):
                jm1, jp1 = max(j - 1, 0), min(j + 1, ny - 1)
                fls0 = target[:, j, i] - field[:, j, i]
                fls1 = target[:, jm1, i] - field[:, jm1, i]
                fls2 = target[:, jp1, i] - field[:, jp1, i]
                fls3 = target[:, j, i - sgn] - field[:, j, i - sgn]
                fls4 = target[:, j, i + sgn] - field[:, j, i + sgn]
                tend[:, j, i] += f * fls0 - g * (fls1 + fls2 + fls3 + fls4 - 4.0 * fls0)
    return tend


def _wrf_weights(dt_full):
    # lbc_fcx_gcx: linear taper over loop=2..relax_zone, fcx=0.1/dt, gcx=fcx/5.
    fcx, gcx = {}, {}
    for b_dist in range(CFG.spec_zone, CFG.relax_zone):
        loop = b_dist + 1
        linear = max(0.0, (CFG.spec_zone + CFG.relax_zone - loop) / (CFG.relax_zone - 1))
        fcx[b_dist] = 0.1 / dt_full * linear
        gcx[b_dist] = 1.0 / dt_full / 50.0 * linear
    return fcx, gcx


def _ring_target_np(leaf, z_len, ny, nx):
    """Scatter a (side, width, z, side_len) strip into a full-ring numpy field."""

    out = np.zeros((z_len, ny, nx))
    width = CFG.spec_zone + CFG.relax_zone
    for b in range(width):
        out[:, :, b] = np.asarray(leaf[0, b, :z_len, :ny])
        out[:, :, nx - 1 - b] = np.asarray(leaf[1, b, :z_len, :ny])
        out[:, b, :] = np.asarray(leaf[2, b, :z_len, :nx])
        out[:, ny - 1 - b, :] = np.asarray(leaf[3, b, :z_len, :nx])
    return out


def test_specified_relax_matches_independent_wrf_stencil():
    rng = np.random.default_rng(20260611)
    nz, ny, nx = 3, 14, 12
    dt_full = 18.0

    mu_total = 1000.0 + rng.normal(size=(ny, nx))
    reference = SimpleNamespace(
        u=jnp.asarray(rng.normal(size=(nz, ny, nx + 1))),
        v=jnp.asarray(rng.normal(size=(nz, ny + 1, nx))),
        theta=jnp.asarray(300.0 + rng.normal(size=(nz, ny, nx))),
        ph_perturbation=jnp.asarray(rng.normal(size=(nz + 1, ny, nx)) * 10.0),
        mu_total=jnp.asarray(mu_total),
        mu_perturbation=jnp.asarray(rng.normal(size=(ny, nx))),
        u_bdy=_leaf(rng, 5, nz, max(ny, nx + 1)),
        v_bdy=_leaf(rng, 5, nz, max(ny + 1, nx)),
        theta_bdy=_leaf(rng, 5, nz, max(ny, nx)) + 300.0,
        ph_bdy=_leaf(rng, 5, nz + 1, max(ny, nx), scale=10.0),
        mu_bdy=_leaf(rng, 5, 1, max(ny, nx)),
    )
    metrics = SimpleNamespace(
        c1h=jnp.asarray(0.9 + 0.01 * np.arange(nz)),
        c2h=jnp.asarray(50.0 + np.arange(nz)),
        c1f=jnp.asarray(0.9 + 0.01 * np.arange(nz + 1)),
        c2f=jnp.asarray(50.0 + np.arange(nz + 1)),
        msfuy=jnp.asarray(1.0 + 0.001 * rng.normal(size=(ny, nx + 1))),
        msfvx=jnp.asarray(1.0 + 0.001 * rng.normal(size=(ny + 1, nx))),
        msfty=jnp.asarray(1.0 + 0.001 * rng.normal(size=(ny, nx))),
    )

    bundle = specified_relax_dry_tendencies(reference, 0.0, metrics, dt_full, CFG)
    fcx, gcx = _wrf_weights(dt_full)

    # theta: mass-coupled both sides with the reference mass, then /msfty.
    c1h = np.asarray(metrics.c1h)[:, None, None]
    c2h = np.asarray(metrics.c2h)[:, None, None]
    mass_h = c1h * mu_total[None, :, :] + c2h
    th_target = _ring_target_np(np.asarray(reference.theta_bdy[0]), nz, ny, nx)
    expected_t = _wrf_relax_mirror_3d(
        mass_h * np.asarray(reference.theta),
        mass_h * th_target,
        fcx,
        gcx,
        CFG.spec_zone,
        CFG.relax_zone,
    ) / np.asarray(metrics.msfty)[None, :, :]
    np.testing.assert_allclose(np.asarray(bundle.t), expected_t, rtol=1e-12, atol=1e-12)

    # mu: plain 2-D field, no coupling.
    mu_target = _ring_target_np(np.asarray(reference.mu_bdy[0]), 1, ny, nx)
    expected_mu = _wrf_relax_mirror_3d(
        np.asarray(reference.mu_perturbation)[None, :, :],
        mu_target,
        fcx,
        gcx,
        CFG.spec_zone,
        CFG.relax_zone,
    )[0]
    np.testing.assert_allclose(np.asarray(bundle.mu), expected_mu, rtol=1e-12, atol=1e-12)

    # relax tendencies must be zero in ring 0 and beyond the relax zone.
    width = CFG.spec_zone + CFG.relax_zone
    assert np.all(np.asarray(bundle.t)[:, 0, :] == 0.0)
    assert np.all(np.asarray(bundle.t)[:, width:-width, width:-width] == 0.0)


def test_ring0_theta_work_pin_reconstructs_leaf_theta():
    rng = np.random.default_rng(7)
    nz, ny, nx = 3, 8, 9
    c1h = 0.95 + 0.01 * np.arange(nz)
    c2h = 40.0 + np.arange(nz)
    mub = 980.0 + rng.normal(size=(ny, nx))
    mu_cur = rng.normal(size=(ny, nx))
    mu_pin = rng.normal(size=(ny, nx))
    mut = mub + mu_cur
    muts_pin = mub + mu_pin
    t_save = rng.normal(size=(nz, ny, nx))  # theta' at stage entry
    theta_leaf = 300.0 + rng.normal(size=(nz, ny, nx))

    mass_pin = c1h[:, None, None] * muts_pin[None, :, :] + c2h[:, None, None]
    mass_cur = c1h[:, None, None] * mut[None, :, :] + c2h[:, None, None]
    # the production pin (operational_mode._acoustic_core_state_from_prep)
    theta_work_pin = mass_pin * (theta_leaf - 300.0) - mass_cur * t_save

    # small_step_finish reconstruction with the ring muts pinned to muts_pin:
    theta_pert = (theta_work_pin + t_save * mass_cur) / mass_pin
    np.testing.assert_allclose(theta_pert, theta_leaf - 300.0, rtol=1e-12)


def test_tangential_work_targets_reconstruct_leaf_winds():
    rng = np.random.default_rng(11)
    nz, ny, nx = 3, 8, 9
    u_save = rng.normal(size=(nz, ny, nx + 1))
    mass_cur = 900.0 + rng.normal(size=(nz, ny, nx + 1))
    mass_stage = 905.0 + rng.normal(size=(nz, ny, nx + 1))
    msfuy = 1.0 + 0.001 * rng.normal(size=(ny, nx + 1))
    u_leaf = _leaf(rng, 5, nz, max(ny, nx + 1), scale=5.0)

    target = tangential_bdy_work_target_u(
        u_leaf[0], jnp.asarray(u_save), jnp.asarray(mass_cur), jnp.asarray(mass_stage), jnp.asarray(msfuy), config=CFG
    )
    # finish: u = (msf*work + save*mass_cur)/mass_stage == leaf on the S/N rows.
    for row, side in ((0, 2), (ny - 1, 3)):
        rebuilt = (msfuy[row] * np.asarray(target)[:, row, :] + u_save[:, row, :] * mass_cur[:, row, :]) / mass_stage[:, row, :]
        np.testing.assert_allclose(rebuilt, np.asarray(u_leaf[0, side, 0, :nz, : nx + 1]), rtol=1e-12)

    v_save = rng.normal(size=(nz, ny + 1, nx))
    mass_v_cur = 900.0 + rng.normal(size=(nz, ny + 1, nx))
    mass_v_stage = 905.0 + rng.normal(size=(nz, ny + 1, nx))
    msfvx = 1.0 + 0.001 * rng.normal(size=(ny + 1, nx))
    v_leaf = _leaf(rng, 5, nz, max(ny + 1, nx), scale=5.0)
    target_v = tangential_bdy_work_target_v(
        v_leaf[0], jnp.asarray(v_save), jnp.asarray(mass_v_cur), jnp.asarray(mass_v_stage), jnp.asarray(msfvx), config=CFG
    )
    for col, side in ((0, 0), (nx - 1, 1)):
        rebuilt = (
            msfvx[:, col] * np.asarray(target_v)[:, :, col] + v_save[:, :, col] * mass_v_cur[:, :, col]
        ) / mass_v_stage[:, :, col]
        np.testing.assert_allclose(rebuilt, np.asarray(v_leaf[0, side, 0, :nz, : ny + 1]), rtol=1e-12)


def _wrf_flux_face_mirror(field, vel, m, upstream):
    """Literal WRF order-5 degraded flux at face ``m`` (1-D arrays).

    module_advect_em.F flux operators + the degrade tiers; face m sits between
    cells m-1 and m, ``vel[m]`` is the transporting velocity at the face.
    """

    n = field.shape[0]
    q = field
    v = vel[m]
    if 3 <= m <= n - 3:
        f6 = (37.0 * (q[m] + q[m - 1]) - 8.0 * (q[m + 1] + q[m - 2]) + (q[m + 2] + q[m - 3])) / 60.0
        c6 = ((q[m + 2] - q[m - 3]) - 5.0 * (q[m + 1] - q[m - 2]) + 10.0 * (q[m] - q[m - 1])) / 60.0
        return v * (f6 - np.sign(v) * c6)
    if m in (2, n - 2):
        f4 = (7.0 * (q[m] + q[m - 1]) - (q[m + 1] + q[m - 2])) / 12.0
        c4 = ((q[m + 1] - q[m - 2]) - 3.0 * (q[m] - q[m - 1])) / 12.0
        return v * (f4 + np.sign(v) * c4)
    if m == 1:
        qb = q[0]
        if upstream and q[1] < 0.0:
            qb = q[1]
        return v * 0.5 * (q[1] + qb)
    if m == n - 1:
        qb = q[n - 1]
        if upstream and q[n - 2] > 0.0:
            qb = q[n - 2]
        return v * 0.5 * (q[n - 2] + qb)
    return 0.0


def test_specified_flux_faces_match_wrf_tiers():
    rng = np.random.default_rng(3)
    n = 16
    field = rng.normal(size=(1, 1, n))
    vel = rng.normal(size=(1, 1, n))
    for upstream in (False, True):
        faces = np.asarray(specified_flux_faces(jnp.asarray(field), jnp.asarray(vel), 2, upstream=upstream))
        for m in range(n):
            expected = _wrf_flux_face_mirror(field[0, 0], vel[0, 0], m, upstream)
            np.testing.assert_allclose(faces[0, 0, m], expected, rtol=1e-13, atol=1e-15,
                                       err_msg=f"face {m} upstream={upstream}")


def test_specified_div_masks_outer_cells():
    rng = np.random.default_rng(4)
    n = 12
    fq = jnp.asarray(rng.normal(size=(2, 3, n)))
    div = np.asarray(_specified_div(fq, 2))
    fq_np = np.asarray(fq)
    assert np.all(div[:, :, 0] == 0.0)
    assert np.all(div[:, :, n - 1] == 0.0)
    np.testing.assert_allclose(div[:, :, 1 : n - 1], fq_np[:, :, 2:n] - fq_np[:, :, 1 : n - 1], rtol=1e-15)


def _boundary(grid: GridSpec, z_len: int, value: float, dtype=jnp.float32):
    side = max(grid.nx + 1, grid.ny + 1)
    return jnp.ones((2, 4, z_len, side), dtype=dtype) * value


def test_dry_spec_only_pins_ring0_keeps_p_and_moisture():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid).replace(
        theta=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float32),
        qv=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float32),
        u=jnp.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=jnp.float32),
        v=jnp.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=jnp.float32),
        w=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        p_total=jnp.ones((grid.nz, grid.ny, grid.nx), dtype=jnp.float64) * 1000.0,
        p_perturbation=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        ph=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        ph_total=jnp.ones((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64) * 100.0,
        ph_perturbation=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        mu=jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
        mu_total=jnp.ones((grid.ny, grid.nx), dtype=jnp.float64) * 1000.0,
        mu_perturbation=jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
        theta_bdy=_boundary(grid, grid.nz, 10.0),
        qv_bdy=_boundary(grid, grid.nz, 0.002),
        u_bdy=_boundary(grid, grid.nz, 3.0),
        v_bdy=_boundary(grid, grid.nz, 4.0),
        w_bdy=_boundary(grid, grid.nz + 1, 7.0, dtype=jnp.float64),
        p_bdy=_boundary(grid, grid.nz, 8.0, dtype=jnp.float64),
        pb_bdy=_boundary(grid, grid.nz, 900.0, dtype=jnp.float64),
        ph_bdy=_boundary(grid, grid.nz + 1, 5.0, dtype=jnp.float64),
        phb_bdy=_boundary(grid, grid.nz + 1, 95.0, dtype=jnp.float64),
        mu_bdy=_boundary(grid, 1, 6.0, dtype=jnp.float64),
        mub_bdy=_boundary(grid, 1, 990.0, dtype=jnp.float64),
    )

    out = apply_lateral_boundaries(state, 0.0, 60.0, dry_spec_only=True)

    # ring 0 spec set for the dry fields (full theta strip = 10).
    assert np.allclose(np.asarray(out.theta[:, 1:-1, 0]), 10.0)
    assert np.allclose(np.asarray(out.w[:, 1:-1, 0]), 7.0)
    assert np.allclose(np.asarray(out.mu_perturbation[1:-1, 0]), 6.0)
    assert np.allclose(np.asarray(out.ph_perturbation[:, 1:-1, 0]), 5.0)
    # NO relax-zone write for dry fields: ring 1 stays at the interior value.
    assert np.allclose(np.asarray(out.theta[:, 3:5, 1]), 0.0)
    assert np.allclose(np.asarray(out.ph_perturbation[:, 3:5, 1]), 0.0)
    # p'/pb NEVER forced (WRF does not force the diagnostic pressure).
    assert np.allclose(np.asarray(out.p_perturbation), 0.0)
    assert np.allclose(np.asarray(out.p_total), 1000.0)
    # moisture keeps the FULL spec+relax treatment.
    assert np.allclose(np.asarray(out.qv[:, 1:-1, 0]), 0.002)
    assert np.asarray(out.qv[:, 3:5, 1]).max() > 0.0
    assert np.all(np.asarray(out.qv) >= 0.0)
