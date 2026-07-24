"""CPU/oracle gates for NESTED_FROZEN_WRF_BOUNDARY_BUNDLE_V1.

Every numerical expectation below is independent algebra from pristine WRF
v4.7.1 ``module_bc_em.F`` / ``share/module_bc.F``.  No GPU is touched.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.boundary_apply import (
    BoundaryConfig,
    apply_lateral_boundaries,
    apply_normal_bdy_work,
    interpolate_boundary_leaf,
    normal_bdy_work_target_u,
    normal_bdy_work_target_v,
    specified_relax_dry_tendencies,
    tangential_bdy_work_target_u,
    tangential_bdy_work_target_v,
)
from gpuwrf.runtime.domain_tree import (
    DomainTree,
    _operational_force,
    build_live_nested_boundary_config,
)
from gpuwrf.nesting.boundary_construction import field_sides_3d
from gpuwrf.runtime.aot_cheap_key import static_config_hash
from gpuwrf.integration.nested_pipeline import _make_namelist
from gpuwrf.runtime.operational_mode import (
    _initial_carry_for_run,
    _nested_frozen_wrf_boundary_active,
    _physics_boundary_step,
    _rk_scan_step,
    _stage_entry_mudf,
    nested_boundary_package_endpoint_seconds,
    nested_boundary_stage_seconds,
)
from gpuwrf.validation.moving_nest_testbed import (
    build_flat_grid,
    build_nested_pair,
    build_neutral_state,
)


CFG = BoundaryConfig(
    update_cadence_s=6.0,
    force_geopotential=False,
    nested_frozen_wrf_boundary_bundle=True,
)


def _constant_leaf(value: float, *, z: int, side_len: int, dtype=np.float64):
    return jnp.asarray(
        np.full((2, 4, 5, z, side_len), value, dtype=dtype)
    )


def _random_leaf(rng, *, z: int, side_len: int, scale: float = 1.0):
    return jnp.asarray(rng.normal(size=(2, 4, 5, z, side_len)) * scale)


def _ring_target(leaf, *, z: int, ny: int, nx: int):
    """Numpy WRF side registration: S/N own all corners."""

    leaf = np.asarray(leaf)
    out = np.zeros((z, ny, nx), dtype=leaf.dtype)
    for b in range(5):
        out[:, :, b] = leaf[0, b, :z, :ny]
        out[:, :, nx - 1 - b] = leaf[1, b, :z, :ny]
        out[:, b, :] = leaf[2, b, :z, :nx]
        out[:, ny - 1 - b, :] = leaf[3, b, :z, :nx]
    return out


def _wrf_relax(field, target, *, dt: float, spec_zone: int = 1, relax_zone: int = 4):
    """Literal ``relax_bdytend_core`` for one coupled 3-D field."""

    field = np.asarray(field)
    target = np.asarray(target)
    z, ny, nx = field.shape
    out = np.zeros_like(field)
    for b in range(spec_zone, relax_zone):
        loop = b + 1
        linear = (spec_zone + relax_zone - loop) / (relax_zone - 1)
        fcx = 0.1 / dt * linear
        gcx = 1.0 / dt / 50.0 * linear
        # Y sides own the diagonal/corners.
        for row, inward in ((b, 1), (ny - 1 - b, -1)):
            for i in range(b, nx - b):
                im1, ip1 = max(i - 1, 0), min(i + 1, nx - 1)
                residual = target - field
                fls0 = residual[:, row, i]
                lap = (
                    residual[:, row, im1]
                    + residual[:, row, ip1]
                    + residual[:, row - inward, i]
                    + residual[:, row + inward, i]
                    - 4.0 * fls0
                )
                out[:, row, i] += fcx * fls0 - gcx * lap
        # X sides exclude cells already owned by Y.
        for col, inward in ((b, 1), (nx - 1 - b, -1)):
            for j in range(b + 1, ny - 1 - b):
                jm1, jp1 = max(j - 1, 0), min(j + 1, ny - 1)
                residual = target - field
                fls0 = residual[:, j, col]
                lap = (
                    residual[:, jm1, col]
                    + residual[:, jp1, col]
                    + residual[:, j, col - inward]
                    + residual[:, j, col + inward]
                    - 4.0 * fls0
                )
                out[:, j, col] += fcx * fls0 - gcx * lap
    return out


def _relax_fixture(*, random: bool):
    rng = np.random.default_rng(234)
    nz, ny, nx = 3, 14, 13
    side = max(ny + 1, nx + 1)
    if random:
        mu_total = 900.0 + rng.normal(size=(ny, nx))
        u = rng.normal(size=(nz, ny, nx + 1))
        v = rng.normal(size=(nz, ny + 1, nx))
        w = rng.normal(size=(nz + 1, ny, nx))
        theta = 300.0 + rng.normal(size=(nz, ny, nx))
        ph = rng.normal(size=(nz + 1, ny, nx))
        mu_p = rng.normal(size=(ny, nx))
        u_bdy = _random_leaf(rng, z=nz, side_len=side)
        v_bdy = _random_leaf(rng, z=nz, side_len=side)
        w_bdy = _random_leaf(rng, z=nz + 1, side_len=side)
        theta_bdy = 300.0 + _random_leaf(rng, z=nz, side_len=side)
        ph_bdy = _random_leaf(rng, z=nz + 1, side_len=side)
        mu_bdy = _random_leaf(rng, z=1, side_len=side)
        msfuy = 1.0 + 0.01 * rng.random((ny, nx + 1))
        msfvx = 1.0 + 0.01 * rng.random((ny + 1, nx))
        msfty = 1.0 + 0.01 * rng.random((ny, nx))
    else:
        mu_total = np.full((ny, nx), 1000.0)
        u = np.full((nz, ny, nx + 1), 2.0)
        v = np.full((nz, ny + 1, nx), 2.0)
        w = np.full((nz + 1, ny, nx), 2.0)
        theta = np.full((nz, ny, nx), 302.0)
        ph = np.full((nz + 1, ny, nx), 2.0)
        mu_p = np.full((ny, nx), 2.0)
        u_bdy = _constant_leaf(2.0, z=nz, side_len=side)
        v_bdy = _constant_leaf(2.0, z=nz, side_len=side)
        w_bdy = _constant_leaf(2.0, z=nz + 1, side_len=side)
        theta_bdy = _constant_leaf(302.0, z=nz, side_len=side)
        ph_bdy = _constant_leaf(2.0, z=nz + 1, side_len=side)
        mu_bdy = _constant_leaf(2.0, z=1, side_len=side)
        msfuy = np.ones((ny, nx + 1))
        msfvx = np.ones((ny + 1, nx))
        msfty = np.ones((ny, nx))
    reference = SimpleNamespace(
        u=jnp.asarray(u),
        v=jnp.asarray(v),
        w=jnp.asarray(w),
        theta=jnp.asarray(theta),
        ph_perturbation=jnp.asarray(ph),
        mu_total=jnp.asarray(mu_total),
        mu_perturbation=jnp.asarray(mu_p),
        u_bdy=u_bdy,
        v_bdy=v_bdy,
        w_bdy=w_bdy,
        theta_bdy=theta_bdy,
        ph_bdy=ph_bdy,
        mu_bdy=mu_bdy,
    )
    metrics = SimpleNamespace(
        c1h=jnp.asarray(0.9 + 0.01 * np.arange(nz)),
        c2h=jnp.asarray(40.0 + np.arange(nz)),
        c1f=jnp.asarray(0.9 + 0.01 * np.arange(nz + 1)),
        c2f=jnp.asarray(40.0 + np.arange(nz + 1)),
        msfuy=jnp.asarray(msfuy),
        msfvx=jnp.asarray(msfvx),
        msfty=jnp.asarray(msfty),
    )
    return reference, metrics


def test_two_leaf_absolute_saturation_and_integer_subcycle_repair():
    leaf = jnp.stack(
        (jnp.zeros((4, 5, 1, 1)), jnp.ones((4, 5, 1, 1))), axis=0
    )
    absolute = [
        float(interpolate_boundary_leaf(leaf, step * 2.0, 6.0)[0, 0, 0, 0])
        for step in range(1, 7)
    ]
    assert absolute == [1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0]

    for child_dt, parent_dt in ((6.0, 18.0), (2.0, 6.0), (1.5, 7.5)):
        ratio = int(parent_dt / child_dt)
        got = np.asarray(
            nested_boundary_package_endpoint_seconds(
                jnp.arange(1, 2 * ratio + 1),
                child_dt_s=child_dt,
                parent_cadence_s=parent_dt,
            )
        )
        one_cycle = child_dt * np.arange(1, ratio + 1)
        np.testing.assert_array_equal(got, np.tile(one_cycle, 2))
        assert got[ratio - 1] == parent_dt
        assert got[ratio] == child_dt


def test_stage_clock_walks_from_child_start_to_package_endpoint():
    endpoint = 6.0
    got = [
        float(
            nested_boundary_stage_seconds(
                endpoint, child_dt_s=2.0, rk_dt_s=rk_dt
            )
        )
        for rk_dt in (2.0 / 3.0, 1.0, 2.0)
    ]
    np.testing.assert_allclose(got, [14.0 / 3.0, 5.0, 6.0], rtol=0, atol=1e-15)


def test_nonintegral_parent_ratio_fails_closed():
    try:
        nested_boundary_package_endpoint_seconds(
            1, child_dt_s=2.0, parent_cadence_s=7.0
        )
    except ValueError as error:
        assert "integer child-dt multiple" in str(error)
    else:
        raise AssertionError("nonintegral subcycle ratio must fail closed")


def test_full_nested_relax_zero_residual_is_exact_identity():
    reference, metrics = _relax_fixture(random=False)
    bundle = specified_relax_dry_tendencies(
        reference, 6.0, metrics, 2.0, CFG, include_nested_w=True
    )
    for name in ("ru", "rv", "t", "ph", "mu", "w"):
        value = getattr(bundle, name)
        assert value is not None
        assert np.count_nonzero(np.asarray(value)) == 0, name


def test_full_nested_relax_matches_independent_wrf_mass_map_oracle():
    reference, metrics = _relax_fixture(random=True)
    dt = 2.0
    bundle = specified_relax_dry_tendencies(
        reference, 6.0, metrics, dt, CFG, include_nested_w=True
    )
    nz, ny, nx = reference.theta.shape
    mu = np.asarray(reference.mu_total)
    c1h = np.asarray(metrics.c1h)[:, None, None]
    c2h = np.asarray(metrics.c2h)[:, None, None]
    c1f = np.asarray(metrics.c1f)[:, None, None]
    c2f = np.asarray(metrics.c2f)[:, None, None]
    mass_h = c1h * mu[None] + c2h
    mass_f = c1f * mu[None] + c2f
    muu = 0.5 * (
        np.concatenate((mu[:, :1], mu), axis=1)
        + np.concatenate((mu, mu[:, -1:]), axis=1)
    )
    muv = 0.5 * (
        np.concatenate((mu[:1, :], mu), axis=0)
        + np.concatenate((mu, mu[-1:, :]), axis=0)
    )
    mass_u = c1h * muu[None] + c2h
    mass_v = c1h * muv[None] + c2h
    u_target = _ring_target(reference.u_bdy[1], z=nz, ny=ny, nx=nx + 1)
    v_target = _ring_target(reference.v_bdy[1], z=nz, ny=ny + 1, nx=nx)
    t_target = _ring_target(reference.theta_bdy[1], z=nz, ny=ny, nx=nx)
    ph_target = _ring_target(reference.ph_bdy[1], z=nz + 1, ny=ny, nx=nx)
    w_target = _ring_target(reference.w_bdy[1], z=nz + 1, ny=ny, nx=nx)
    mu_target = _ring_target(reference.mu_bdy[1], z=1, ny=ny, nx=nx)
    expected = {
        "ru": _wrf_relax(
            mass_u * np.asarray(reference.u) / np.asarray(metrics.msfuy)[None],
            mass_u * u_target / np.asarray(metrics.msfuy)[None],
            dt=dt,
        ),
        "rv": _wrf_relax(
            mass_v * np.asarray(reference.v) / np.asarray(metrics.msfvx)[None],
            mass_v * v_target / np.asarray(metrics.msfvx)[None],
            dt=dt,
        ),
        "t": _wrf_relax(
            mass_h * np.asarray(reference.theta), mass_h * t_target, dt=dt
        )
        / np.asarray(metrics.msfty)[None],
        "ph": _wrf_relax(
            mass_f * np.asarray(reference.ph_perturbation),
            mass_f * ph_target,
            dt=dt,
        )
        / np.asarray(metrics.msfty)[None],
        "w": _wrf_relax(
            mass_f * np.asarray(reference.w), mass_f * w_target, dt=dt
        )
        / np.asarray(metrics.msfty)[None],
        "mu": _wrf_relax(
            np.asarray(reference.mu_perturbation)[None], mu_target, dt=dt
        )[0],
    }
    for name, oracle in expected.items():
        np.testing.assert_allclose(
            np.asarray(getattr(bundle, name)), oracle, rtol=2e-13, atol=2e-13
        )


def test_coupled_forcedown_relax_consumes_records_without_second_mass_weight():
    """Exact nest records are already coupled by med_nest_force."""

    reference, metrics = _relax_fixture(random=True)
    rng = np.random.default_rng(471)
    nz, ny, nx = reference.theta.shape
    side = int(reference.u_bdy.shape[-1])
    reference = SimpleNamespace(
        **{
            **vars(reference),
            "u_bdy": _random_leaf(rng, z=nz, side_len=side, scale=800.0),
            "v_bdy": _random_leaf(rng, z=nz, side_len=side, scale=800.0),
            "w_bdy": _random_leaf(rng, z=nz + 1, side_len=side, scale=800.0),
            "theta_bdy": _random_leaf(rng, z=nz, side_len=side, scale=800.0),
            "ph_bdy": _random_leaf(rng, z=nz + 1, side_len=side, scale=800.0),
        }
    )
    dt = 2.0
    bundle = specified_relax_dry_tendencies(
        reference,
        6.0,
        metrics,
        dt,
        CFG,
        include_nested_w=True,
        coupled_boundary_leaves=True,
    )
    mu = np.asarray(reference.mu_total)
    c1h = np.asarray(metrics.c1h)[:, None, None]
    c2h = np.asarray(metrics.c2h)[:, None, None]
    c1f = np.asarray(metrics.c1f)[:, None, None]
    c2f = np.asarray(metrics.c2f)[:, None, None]
    mass_h = c1h * mu[None] + c2h
    mass_f = c1f * mu[None] + c2f
    muu = 0.5 * (
        np.concatenate((mu[:, :1], mu), axis=1)
        + np.concatenate((mu, mu[:, -1:]), axis=1)
    )
    muv = 0.5 * (
        np.concatenate((mu[:1, :], mu), axis=0)
        + np.concatenate((mu, mu[-1:, :]), axis=0)
    )
    current = {
        "ru": (c1h * muu[None] + c2h)
        * np.asarray(reference.u)
        / np.asarray(metrics.msfuy)[None],
        "rv": (c1h * muv[None] + c2h)
        * np.asarray(reference.v)
        / np.asarray(metrics.msfvx)[None],
        "t": mass_h * (np.asarray(reference.theta) - 300.0),
        "ph": mass_f * np.asarray(reference.ph_perturbation),
        # WRF relax_bdy_dry mass_weight(w) deliberately has no msfty here.
        "w": mass_f * np.asarray(reference.w),
    }
    targets = {
        "ru": _ring_target(reference.u_bdy[1], z=nz, ny=ny, nx=nx + 1),
        "rv": _ring_target(reference.v_bdy[1], z=nz, ny=ny + 1, nx=nx),
        "t": _ring_target(reference.theta_bdy[1], z=nz, ny=ny, nx=nx),
        "ph": _ring_target(reference.ph_bdy[1], z=nz + 1, ny=ny, nx=nx),
        "w": _ring_target(reference.w_bdy[1], z=nz + 1, ny=ny, nx=nx),
    }
    for name in ("ru", "rv", "t", "ph", "w"):
        expected = _wrf_relax(current[name], targets[name], dt=dt)
        if name in ("t", "ph", "w"):
            expected = expected / np.asarray(metrics.msfty)[None]
        np.testing.assert_allclose(
            np.asarray(getattr(bundle, name)), expected, rtol=3e-13, atol=3e-13
        )


def test_uniform_residual_has_exact_fcx_response_and_no_interior_source():
    """Uniform residual kills gcx and leaves the literal WRF fcx taper."""

    reference, metrics = _relax_fixture(random=False)
    side = int(reference.u_bdy.shape[-1])
    reference = SimpleNamespace(
        **{
            **vars(reference),
            "u_bdy": _constant_leaf(3.0, z=reference.u.shape[0], side_len=side),
            "v_bdy": _constant_leaf(3.0, z=reference.v.shape[0], side_len=side),
            "w_bdy": _constant_leaf(3.0, z=reference.w.shape[0], side_len=side),
            "theta_bdy": _constant_leaf(
                303.0, z=reference.theta.shape[0], side_len=side
            ),
            "ph_bdy": _constant_leaf(
                3.0, z=reference.ph_perturbation.shape[0], side_len=side
            ),
            "mu_bdy": _constant_leaf(3.0, z=1, side_len=side),
        }
    )
    dt = 2.0
    bundle = specified_relax_dry_tendencies(
        reference, 6.0, metrics, dt, CFG, include_nested_w=True
    )
    arrays = {name: np.asarray(getattr(bundle, name)) for name in ("ru", "rv", "t", "ph", "mu", "w")}

    mu = np.asarray(reference.mu_total)
    c1h = np.asarray(metrics.c1h)[:, None, None]
    c2h = np.asarray(metrics.c2h)[:, None, None]
    c1f = np.asarray(metrics.c1f)[:, None, None]
    c2f = np.asarray(metrics.c2f)[:, None, None]
    muu = 0.5 * (
        np.concatenate((mu[:, :1], mu), axis=1)
        + np.concatenate((mu, mu[:, -1:]), axis=1)
    )
    muv = 0.5 * (
        np.concatenate((mu[:1, :], mu), axis=0)
        + np.concatenate((mu, mu[-1:, :]), axis=0)
    )
    residual_scale = {
        "ru": (c1h * muu[None] + c2h) / np.asarray(metrics.msfuy)[None],
        "rv": (c1h * muv[None] + c2h) / np.asarray(metrics.msfvx)[None],
        "t": (c1h * mu[None] + c2h) / np.asarray(metrics.msfty)[None],
        "ph": (c1f * mu[None] + c2f) / np.asarray(metrics.msfty)[None],
        "w": (c1f * mu[None] + c2f) / np.asarray(metrics.msfty)[None],
        "mu": np.ones_like(reference.mu_perturbation)[None],
    }
    # Rows b=1,2,3 integrate to 0.1, 0.1*2/3, 0.1/3 of the
    # coupled residual over one full step.  The returned bundle is per second.
    for b, linear in ((1, 1.0), (2, 2.0 / 3.0), (3, 1.0 / 3.0)):
        for name, value in arrays.items():
            value3 = value if value.ndim == 3 else value[None]
            scale = residual_scale[name]
            if name == "mu":
                scale = scale[0]
            expected = (0.1 / dt) * linear * scale[..., b, b]
            np.testing.assert_allclose(
                value3[..., b, b], expected, rtol=2e-13, atol=2e-13
            )
    for value in arrays.values():
        value3 = value if value.ndim == 3 else value[None]
        assert np.count_nonzero(value3[..., 4:-4, 4:-4]) == 0


def test_nested_w_is_nested_only_and_frozen_bundle_is_hoisted_once():
    reference, metrics = _relax_fixture(random=False)
    root = specified_relax_dry_tendencies(reference, 0.0, metrics, 2.0, CFG)
    nested = specified_relax_dry_tendencies(
        reference, 0.0, metrics, 2.0, CFG, include_nested_w=True
    )
    assert root.w is None
    assert nested.w is not None
    source = inspect.getsource(_rk_scan_step)
    assert source.count("_nested_frozen_bdy_relax(") == 1
    assert source.index("nested_frozen_relax =") < source.index("def advance_stage")


def test_mudf_zeroes_only_candidate_rk1():
    carried = jnp.arange(12.0).reshape(3, 4)
    np.testing.assert_array_equal(
        np.asarray(
            _stage_entry_mudf(
                carried, rk_step=1, nested_frozen_bundle=True
            )
        ),
        0.0,
    )
    for rk_step, enabled in ((2, True), (3, True), (1, False)):
        np.testing.assert_array_equal(
            np.asarray(
                _stage_entry_mudf(
                    carried, rk_step=rk_step, nested_frozen_bundle=enabled
                )
            ),
            np.asarray(carried),
        )


def test_strength20_negative_control_and_candidate_retires_relax_rows():
    u0 = jnp.zeros((1, 14, 15))
    v0 = jnp.zeros((1, 15, 14))
    ut = jnp.ones_like(u0)
    vt = jnp.ones_like(v0)
    released_u, released_v = u0, v0
    candidate_u, candidate_v = u0, v0
    for _ in range(10):
        released_u, released_v = apply_normal_bdy_work(
            released_u, released_v, ut, vt, 0.6, 6.0, relax_strength=20.0
        )
        candidate_u, candidate_v = apply_normal_bdy_work(
            candidate_u,
            candidate_v,
            ut,
            vt,
            0.6,
            6.0,
            relax_strength=20.0,
            relax_rows=False,
            wrf_single_owner=True,
        )
    assert abs(float(released_u[0, 7, 1]) - 0.8926258175999999) < 2.0e-16
    assert float(candidate_u[0, 7, 1]) == 0.0
    assert float(candidate_v[0, 1, 7]) == 0.0
    assert float(candidate_u[0, 7, 0]) == 1.0
    assert float(candidate_v[0, 0, 7]) == 1.0
    # Candidate normal-u does not double-write the S/N-owned corners.
    assert float(candidate_u[0, 0, 0]) == 0.0


def test_wind_work_inverse_uses_stage_frozen_finish_mass_exactly():
    """Each work pin reconstructs its boundary velocity under finish algebra."""

    rng = np.random.default_rng(9314)
    nz, ny, nx = 2, 8, 7
    side = max(ny + 1, nx + 1)
    u_save = jnp.asarray(rng.normal(size=(nz, ny, nx + 1)))
    v_save = jnp.asarray(rng.normal(size=(nz, ny + 1, nx)))
    u_cur = jnp.asarray(800.0 + rng.random(size=u_save.shape))
    u_stage = jnp.asarray(900.0 + rng.random(size=u_save.shape))
    v_cur = jnp.asarray(800.0 + rng.random(size=v_save.shape))
    v_stage = jnp.asarray(900.0 + rng.random(size=v_save.shape))
    msfuy = jnp.asarray(1.0 + 0.01 * rng.random(size=(ny, nx + 1)))
    msfvx = jnp.asarray(1.0 + 0.01 * rng.random(size=(ny + 1, nx)))
    u_leaf = _constant_leaf(7.0, z=nz, side_len=side)[1]
    v_leaf = _constant_leaf(-3.0, z=nz, side_len=side)[1]

    u_normal = normal_bdy_work_target_u(
        u_leaf, u_save, u_cur, u_stage, msfuy, config=CFG
    )
    u_tangent = tangential_bdy_work_target_u(
        u_leaf, u_save, u_cur, u_stage, msfuy, config=CFG
    )
    v_normal = normal_bdy_work_target_v(
        v_leaf, v_save, v_cur, v_stage, msfvx, config=CFG
    )
    v_tangent = tangential_bdy_work_target_v(
        v_leaf, v_save, v_cur, v_stage, msfvx, config=CFG
    )
    u_normal_finished = (
        msfuy[None] * u_normal + u_save * u_cur
    ) / u_stage
    u_tangent_finished = (
        msfuy[None] * u_tangent + u_save * u_cur
    ) / u_stage
    v_normal_finished = (
        msfvx[None] * v_normal + v_save * v_cur
    ) / v_stage
    v_tangent_finished = (
        msfvx[None] * v_tangent + v_save * v_cur
    ) / v_stage
    np.testing.assert_allclose(np.asarray(u_normal_finished[:, :, 0]), 7.0)
    np.testing.assert_allclose(np.asarray(u_normal_finished[:, :, -1]), 7.0)
    np.testing.assert_allclose(np.asarray(u_tangent_finished[:, 0, :]), 7.0)
    np.testing.assert_allclose(np.asarray(u_tangent_finished[:, -1, :]), 7.0)
    np.testing.assert_allclose(np.asarray(v_normal_finished[:, 0, :]), -3.0)
    np.testing.assert_allclose(np.asarray(v_normal_finished[:, -1, :]), -3.0)
    np.testing.assert_allclose(np.asarray(v_tangent_finished[:, :, 0]), -3.0)
    np.testing.assert_allclose(np.asarray(v_tangent_finished[:, :, -1]), -3.0)


def test_coupled_wind_work_inverse_lands_on_coupled_record_exactly():
    rng = np.random.default_rng(1471)
    nz, ny, nx = 2, 8, 7
    side = max(ny + 1, nx + 1)
    u_save = jnp.asarray(rng.normal(size=(nz, ny, nx + 1)))
    v_save = jnp.asarray(rng.normal(size=(nz, ny + 1, nx)))
    u_cur = jnp.asarray(800.0 + rng.random(size=u_save.shape))
    u_stage = jnp.asarray(900.0 + rng.random(size=u_save.shape))
    v_cur = jnp.asarray(800.0 + rng.random(size=v_save.shape))
    v_stage = jnp.asarray(900.0 + rng.random(size=v_save.shape))
    msfuy = jnp.asarray(1.0 + 0.01 * rng.random(size=(ny, nx + 1)))
    msfvx = jnp.asarray(1.0 + 0.01 * rng.random(size=(ny + 1, nx)))
    u_coupled = _constant_leaf(7000.0, z=nz, side_len=side)[1]
    v_coupled = _constant_leaf(-3000.0, z=nz, side_len=side)[1]
    uw = normal_bdy_work_target_u(
        u_coupled,
        u_save,
        u_cur,
        u_stage,
        msfuy,
        config=CFG,
        coupled_boundary_leaves=True,
    )
    vw = normal_bdy_work_target_v(
        v_coupled,
        v_save,
        v_cur,
        v_stage,
        msfvx,
        config=CFG,
        coupled_boundary_leaves=True,
    )
    u_finished = (msfuy[None] * uw + u_save * u_cur) / u_stage
    v_finished = (msfvx[None] * vw + v_save * v_cur) / v_stage
    np.testing.assert_allclose(
        np.asarray(u_finished[:, :, 0] * u_stage[:, :, 0] / msfuy[:, 0][None]),
        7000.0,
    )
    np.testing.assert_allclose(
        np.asarray(v_finished[:, 0, :] * v_stage[:, 0, :] / msfvx[0, :][None]),
        -3000.0,
    )


def _ownership_counts(ny: int, nx: int, *, spec_zone=1, relax_zone=4):
    spec = np.zeros((ny, nx), dtype=np.int32)
    relax = np.zeros((ny, nx), dtype=np.int32)
    for b in range(spec_zone):
        spec[b, :] += 1
        spec[ny - 1 - b, :] += 1
        spec[b + 1 : ny - 1 - b, b] += 1
        spec[b + 1 : ny - 1 - b, nx - 1 - b] += 1
    for b in range(spec_zone, relax_zone):
        relax[b, b : nx - b] += 1
        relax[ny - 1 - b, b : nx - b] += 1
        relax[b + 1 : ny - 1 - b, b] += 1
        relax[b + 1 : ny - 1 - b, nx - 1 - b] += 1
    return spec, relax


def test_wrf_side_corner_ownership_is_exactly_once_for_mass_u_v_staggers():
    for shape in ((14, 13), (14, 14), (15, 13)):
        spec, relax = _ownership_counts(*shape)
        assert set(np.unique(spec)) <= {0, 1}
        assert set(np.unique(relax)) <= {0, 1}
        assert np.count_nonzero(spec == 1) == 2 * shape[1] + 2 * (shape[0] - 2)
        assert np.count_nonzero((spec + relax) > 1) == 0


def test_candidate_end_step_is_complete_noop_after_in_rk_scalar_cadence():
    grid = build_flat_grid(nx=14, ny=14, nz=3, dx_m=1000.0)
    state = build_neutral_state(grid)
    side = max(grid.nx + 1, grid.ny + 1)
    p_total = np.asarray(state.p_total) + 123.0
    p_prime = np.asarray(state.p_perturbation) + 123.0
    ph_base = np.asarray(state.ph_total - state.ph_perturbation)
    mu_base = np.asarray(state.mu_total - state.mu_perturbation)
    mass_h = (
        np.asarray(grid.metrics.c1h)[:, None, None]
        * np.asarray(state.mu_total)[None, :, :]
        + np.asarray(grid.metrics.c2h)[:, None, None]
    )
    qv_coupled = jnp.asarray(mass_h * 0.002)
    qv_strip = field_sides_3d(qv_coupled, 5, side)
    state = state.replace(
        p_total=jnp.asarray(p_total),
        p_perturbation=jnp.asarray(p_prime),
        qv=jnp.zeros_like(state.qv),
        u_bdy=_constant_leaf(3.0, z=grid.nz, side_len=side),
        v_bdy=_constant_leaf(4.0, z=grid.nz, side_len=side),
        w_bdy=_constant_leaf(7.0, z=grid.nz + 1, side_len=side),
        theta_bdy=_constant_leaf(301.0, z=grid.nz, side_len=side),
        qv_bdy=jnp.stack((qv_strip, qv_strip), axis=0),
        p_bdy=_constant_leaf(9.9e9, z=grid.nz, side_len=side),
        pb_bdy=_constant_leaf(8.8e9, z=grid.nz, side_len=side),
        ph_bdy=_constant_leaf(5.0, z=grid.nz + 1, side_len=side),
        phb_bdy=_constant_leaf(7.7e9, z=grid.nz + 1, side_len=side),
        mu_bdy=_constant_leaf(6.0, z=1, side_len=side),
        mub_bdy=_constant_leaf(6.6e9, z=1, side_len=side),
    )
    out = apply_lateral_boundaries(
        state, 6.0, 2.0, CFG, grid.metrics, dry_spec_only=True
    )
    np.testing.assert_array_equal(np.asarray(out.p_total), p_total)
    np.testing.assert_array_equal(np.asarray(out.p_perturbation), p_prime)
    np.testing.assert_allclose(
        np.asarray(out.ph_total - out.ph_perturbation), ph_base, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(out.mu_total - out.mu_perturbation), mu_base, rtol=0, atol=0
    )
    for name in ("u", "v", "w", "theta", "ph_total", "ph_perturbation", "mu_total", "mu_perturbation"):
        np.testing.assert_array_equal(
            np.asarray(getattr(out, name)), np.asarray(getattr(state, name))
        )
    # Candidate-on scalar forcing is frozen at RK1 and consumed by the normal
    # scalar RK update.  The end-step boundary call must therefore be a complete
    # no-op: reapplying values here would double force the scalar and would use
    # the wrong (post-RK) mass/cadence.
    assert jax.tree_util.tree_structure(out) == jax.tree_util.tree_structure(state)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(out),
        jax.tree_util.tree_leaves(state),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_candidate_flag_cannot_change_specified_root_end_sync():
    """The config flag alone never opts a specified/root domain into F1--F6."""

    grid = build_flat_grid(nx=14, ny=14, nz=3, dx_m=1000.0)
    state = build_neutral_state(grid)
    side = max(grid.nx + 1, grid.ny + 1)
    state = state.replace(
        w_bdy=_constant_leaf(17.0, z=grid.nz + 1, side_len=side),
        mub_bdy=_constant_leaf(1234.0, z=1, side_len=side),
        phb_bdy=_constant_leaf(5678.0, z=grid.nz + 1, side_len=side),
    )
    root_off = BoundaryConfig(update_cadence_s=6.0, force_geopotential=True)
    root_flag_present = dataclasses.replace(
        root_off, nested_frozen_wrf_boundary_bundle=True
    )
    released = apply_lateral_boundaries(
        state, 6.0, 2.0, root_off, grid.metrics, dry_spec_only=True
    )
    flagged = apply_lateral_boundaries(
        state, 6.0, 2.0, root_flag_present, grid.metrics, dry_spec_only=True
    )
    for lhs, rhs in zip(
        jax.tree_util.tree_leaves(released),
        jax.tree_util.tree_leaves(flagged),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))


def test_domain_config_default_off_and_single_candidate_surface():
    released = build_live_nested_boundary_config(6.0)
    candidate = build_live_nested_boundary_config(
        6.0, nested_frozen_wrf_boundary_bundle=True
    )
    assert released.nested_frozen_wrf_boundary_bundle is False
    assert candidate.nested_frozen_wrf_boundary_bundle is True
    assert candidate.force_geopotential is False
    differences = [
        field.name
        for field in dataclasses.fields(BoundaryConfig)
        if getattr(released, field.name) != getattr(candidate, field.name)
    ]
    assert differences == ["nested_frozen_wrf_boundary_bundle"]


def test_fresh_production_child_defaults_to_accepted_bundle_with_rollback(monkeypatch):
    grid = build_flat_grid(nx=8, ny=8, nz=3, dx_m=1000.0)
    grid = dataclasses.replace(
        grid,
        bc=dataclasses.replace(grid.bc, source="AIFS"),
    )

    def configured(value):
        if value is None:
            monkeypatch.delenv(
                "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE", raising=False
            )
        else:
            monkeypatch.setenv(
                "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE", value
            )
        return _make_namelist(
            grid=grid,
            tendencies=object(),
            metrics=grid.metrics,
            dt_s=2.0,
            parent_dt_s=6.0,
            run_start=datetime(2026, 7, 13, tzinfo=timezone.utc),
            radiation_static=None,
            cu_physics=0,
        )

    fresh_default = configured(None)
    accepted_explicit = configured("1")
    rollback = configured("0")

    assert fresh_default.boundary_config.nested_frozen_wrf_boundary_bundle is True
    assert _nested_frozen_wrf_boundary_active(fresh_default)
    assert static_config_hash(fresh_default) == static_config_hash(accepted_explicit)

    assert rollback.boundary_config.nested_frozen_wrf_boundary_bundle is False
    assert not _nested_frozen_wrf_boundary_active(rollback)
    assert static_config_hash(fresh_default) != static_config_hash(rollback)


def test_production_child_malformed_boundary_bundle_override_fails_closed(monkeypatch):
    grid = build_flat_grid(nx=8, ny=8, nz=3, dx_m=1000.0)
    for value in ("true", " 1 ", "01", ""):
        monkeypatch.setenv("GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE", value)
        namelist = _make_namelist(
            grid=grid,
            tendencies=object(),
            metrics=grid.metrics,
            dt_s=2.0,
            parent_dt_s=6.0,
            run_start=datetime(2026, 7, 13, tzinfo=timezone.utc),
            radiation_static=None,
            cu_physics=0,
        )
        assert namelist.boundary_config.nested_frozen_wrf_boundary_bundle is False


def test_candidate_off_normal_work_jaxpr_and_stablehlo_are_identical():
    u = jnp.zeros((1, 14, 15))
    v = jnp.zeros((1, 15, 14))
    ut = jnp.ones_like(u)
    vt = jnp.ones_like(v)

    def released(a, b, c, d):
        return apply_normal_bdy_work(a, b, c, d, 0.6, 6.0, relax_strength=20.0)

    def explicit_off(a, b, c, d):
        return apply_normal_bdy_work(
            a,
            b,
            c,
            d,
            0.6,
            6.0,
            relax_strength=20.0,
            relax_rows=True,
            wrf_single_owner=False,
        )

    args = (u, v, ut, vt)
    assert str(jax.make_jaxpr(released)(*args)) == str(jax.make_jaxpr(explicit_off)(*args))
    hlo_a = str(jax.jit(released).lower(*args).compiler_ir(dialect="stablehlo"))
    hlo_b = str(jax.jit(explicit_off).lower(*args).compiler_ir(dialect="stablehlo"))
    normalize = lambda text: re.sub(r"module @jit_[^ ]+", "module @jit_FN", text, count=1)
    assert normalize(hlo_a) == normalize(hlo_b)


def test_complete_candidate_on_ordinary_cpu_step_is_finite_and_interface_stable():
    tree = build_nested_pair(
        parent_nx=18,
        parent_ny=18,
        child_nx=14,
        child_ny=14,
        nz=4,
        ratio=3,
        i_start=3,
        j_start=3,
        dt_s=0.6,
    )
    parent_bundle = tree.domains["d01"]
    child_bundle = tree.domains["d02"]
    child_grid = dataclasses.replace(
        child_bundle.grid,
        bc=dataclasses.replace(child_bundle.grid.bc, source="AIFS"),
    )
    child_namelist = dataclasses.replace(
        child_bundle.namelist,
        grid=child_grid,
        acoustic_substeps=1,
        boundary_config=dataclasses.replace(
            child_bundle.namelist.boundary_config,
            nested_frozen_wrf_boundary_bundle=True,
        ),
    )
    released_namelist = dataclasses.replace(
        child_namelist,
        boundary_config=dataclasses.replace(
            child_namelist.boundary_config,
            nested_frozen_wrf_boundary_bundle=False,
        ),
    )
    assert _nested_frozen_wrf_boundary_active(child_namelist)
    assert not _nested_frozen_wrf_boundary_active(released_namelist)
    assert not _nested_frozen_wrf_boundary_active(
        dataclasses.replace(
            child_namelist,
            boundary_config=dataclasses.replace(
                child_namelist.boundary_config,
                force_geopotential=True,
            ),
        )
    )
    assert not _nested_frozen_wrf_boundary_active(
        dataclasses.replace(child_namelist, run_boundary=False)
    )
    # The single static flag participates in the AOT key; a candidate executable
    # therefore cannot alias a released executable even though the carry shape is
    # deliberately unchanged.
    assert static_config_hash(child_namelist) != static_config_hash(released_namelist)
    candidate_bundle = dataclasses.replace(
        child_bundle, grid=child_grid, namelist=child_namelist
    )
    candidate_tree = DomainTree.from_domains(
        tree.hierarchy,
        {"d01": parent_bundle, "d02": candidate_bundle},
        feedback_enabled=False,
    )
    parent = _initial_carry_for_run(parent_bundle.state, parent_bundle.namelist)
    child = _initial_carry_for_run(candidate_bundle.state, child_namelist)
    forced = _operational_force(candidate_tree.edges["d01"][0], parent, child)
    lowered = jax.jit(
        lambda value: _physics_boundary_step(
            value,
            child_namelist,
            jnp.asarray(1, dtype=jnp.int32),
            run_radiation=False,
        )
    ).lower(forced)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    for forbidden in (
        "xla_python_cpu_callback",
        "host_callback",
        "io_callback",
        "outside_compilation",
        "custom_call",
    ):
        assert forbidden not in stablehlo
    out = _physics_boundary_step(
        forced,
        child_namelist,
        jnp.asarray(1, dtype=jnp.int32),
        run_radiation=False,
    )
    jax.block_until_ready(out.state.theta)
    assert jax.tree_util.tree_structure(out) == jax.tree_util.tree_structure(forced)
    assert len(jax.tree_util.tree_leaves(out)) == len(jax.tree_util.tree_leaves(forced))
    for leaf in jax.tree_util.tree_leaves(out):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.floating):
            assert np.all(np.isfinite(array))

    # Short ordinary-dispatch sequence gate: a second completed step from the
    # same forced carry is finite and exactly reproducible without any observer.
    step2_a = _physics_boundary_step(
        out,
        child_namelist,
        jnp.asarray(2, dtype=jnp.int32),
        run_radiation=False,
    )
    step2_b = _physics_boundary_step(
        out,
        child_namelist,
        jnp.asarray(2, dtype=jnp.int32),
        run_radiation=False,
    )
    jax.block_until_ready(step2_a.state.theta)
    for lhs, rhs in zip(
        jax.tree_util.tree_leaves(step2_a),
        jax.tree_util.tree_leaves(step2_b),
        strict=True,
    ):
        a = np.asarray(lhs)
        b = np.asarray(rhs)
        np.testing.assert_array_equal(a, b)
        if np.issubdtype(a.dtype, np.floating):
            assert np.all(np.isfinite(a))
