from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.dynamics.acoustic_wrf import (
    horizontal_pressure_gradient,
    moisture_coupling_factors,
    x_face_pressure_dpn,
    y_face_pressure_dpn,
)
from gpuwrf.dynamics.metrics import flat_metrics_for_grid


def _state(grid: GridSpec) -> State:
    arrays = {field: jnp.zeros(shape, dtype=jnp.float64) for field, shape in _state_field_shapes(grid).items()}
    arrays["theta"] = jnp.ones_like(arrays["theta"]) * 300.0
    arrays["p_total"] = arrays["p"]
    arrays["p_perturbation"] = arrays["p"]
    arrays["ph_total"] = arrays["ph"]
    arrays["ph_perturbation"] = arrays["ph"]
    arrays["mu_total"] = arrays["mu"]
    arrays["mu_perturbation"] = arrays["mu"]
    return State(**arrays)


def _base(state: State, *, pb: jax.Array | None = None, phb: jax.Array | None = None) -> BaseState:
    return BaseState(
        pb=jnp.zeros_like(state.p_total) if pb is None else pb,
        phb=jnp.zeros_like(state.ph_total) if phb is None else phb,
        mub=jnp.ones_like(state.mu_total) * 90000.0,
        t0=jnp.ones_like(state.theta) * 300.0,
        theta_base=jnp.ones_like(state.theta) * 300.0,
    )


def _ones_cq(state: State) -> tuple[jax.Array, jax.Array]:
    return jnp.ones_like(state.u), jnp.ones_like(state.v)


def test_x_pgf_first_three_terms_cancel_hydrostatic_balance():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _state(grid)
    x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
    p = jnp.broadcast_to(x, state.p_perturbation.shape)
    pb = jnp.broadcast_to(2.0 * x, state.p_perturbation.shape)
    ph = jnp.broadcast_to(-8.0 * x, state.ph_perturbation.shape)
    state = state.replace(ph_perturbation=ph)
    al = jnp.ones_like(p) * 3.0
    alt = jnp.ones_like(p) * 2.0
    cqu, cqv = _ones_cq(state)

    _, _, dpx, _ = horizontal_pressure_gradient(
        state,
        _base(state, pb=pb),
        metrics,
        p,
        al,
        alt,
        cqu,
        cqv,
        non_hydrostatic=False,
    )

    assert float(jnp.max(jnp.abs(dpx))) < 1.0e-10


def test_y_pgf_first_three_terms_cancel_hydrostatic_balance():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _state(grid)
    y = jnp.arange(grid.ny, dtype=jnp.float64)[None, :, None]
    p = jnp.broadcast_to(y, state.p_perturbation.shape)
    pb = jnp.broadcast_to(2.0 * y, state.p_perturbation.shape)
    ph = jnp.broadcast_to(-8.0 * y, state.ph_perturbation.shape)
    state = state.replace(ph_perturbation=ph)
    al = jnp.ones_like(p) * 3.0
    alt = jnp.ones_like(p) * 2.0
    cqu, cqv = _ones_cq(state)

    _, _, _, dpy = horizontal_pressure_gradient(
        state,
        _base(state, pb=pb),
        metrics,
        p,
        al,
        alt,
        cqu,
        cqv,
        non_hydrostatic=False,
    )

    assert float(jnp.max(jnp.abs(dpy))) < 1.0e-10


def test_fourth_term_is_zero_without_nonhydrostatic_pressure_or_mu_perturbation():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _state(grid)
    x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
    state = state.replace(ph_perturbation=jnp.broadcast_to(x, state.ph_perturbation.shape))
    p = jnp.zeros_like(state.p_perturbation)
    al = jnp.zeros_like(p)
    alt = jnp.ones_like(p)
    cqu, cqv = _ones_cq(state)

    _, _, dpx_hydro, dpy_hydro = horizontal_pressure_gradient(
        state,
        _base(state),
        metrics,
        p,
        al,
        alt,
        cqu,
        cqv,
        non_hydrostatic=False,
    )
    _, _, dpx_nonhydro, dpy_nonhydro = horizontal_pressure_gradient(
        state,
        _base(state),
        metrics,
        p,
        al,
        alt,
        cqu,
        cqv,
        non_hydrostatic=True,
    )

    assert jnp.allclose(dpx_nonhydro, dpx_hydro)
    assert jnp.allclose(dpy_nonhydro, dpy_hydro)


def test_fourth_term_activates_with_php_gradient_and_mu_perturbation():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _state(grid)
    x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
    state = state.replace(
        ph_perturbation=jnp.broadcast_to(x, state.ph_perturbation.shape),
        mu_perturbation=jnp.ones_like(state.mu_perturbation) * 25.0,
        mu_total=jnp.ones_like(state.mu_total) * 90025.0,
    )
    p = jnp.zeros_like(state.p_perturbation)
    al = jnp.zeros_like(p)
    alt = jnp.ones_like(p)
    cqu, cqv = _ones_cq(state)

    du_dt, _, dpx, _ = horizontal_pressure_gradient(
        state,
        _base(state),
        metrics,
        p,
        al,
        alt,
        cqu,
        cqv,
        non_hydrostatic=True,
    )

    assert float(jnp.max(jnp.abs(dpx[:, :, 1:-1]))) > 0.0
    assert float(jnp.max(jnp.abs(du_dt[:, :, 1:-1]))) > 0.0


def test_dpn_uses_cf_boundary_and_fnm_fnp_interior_coefficients():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    p_profile = jnp.arange(1, grid.nz + 1, dtype=jnp.float64)[:, None, None] * 10.0
    pressure = jnp.broadcast_to(p_profile, (grid.nz, grid.ny, grid.nx))

    dpn_x = x_face_pressure_dpn(pressure, metrics)
    dpn_y = y_face_pressure_dpn(pressure, metrics)

    expected_bottom = metrics.cf1 * pressure[0, 0, 0] + metrics.cf2 * pressure[1, 0, 0] + metrics.cf3 * pressure[2, 0, 0]
    expected_interior = metrics.fnm[1] * pressure[1, 0, 0] + metrics.fnp[1] * pressure[0, 0, 0]
    assert jnp.allclose(dpn_x[0], expected_bottom)
    assert jnp.allclose(dpn_y[0], expected_bottom)
    assert jnp.allclose(dpn_x[1], expected_interior)
    assert jnp.allclose(dpn_y[1], expected_interior)
    assert jnp.allclose(dpn_x[-1], 0.0)
    assert jnp.allclose(dpn_y[-1], 0.0)


def test_metric_ratio_and_cqu_scale_final_u_tendency():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    scaled_metrics = replace(metrics, msfux=metrics.msfux * 2.0)
    state = _state(grid)
    x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
    state = state.replace(ph_perturbation=jnp.broadcast_to(x, state.ph_perturbation.shape))
    moist_state = state.replace(qv=jnp.ones_like(state.qv) * 0.5)
    p = jnp.zeros_like(state.p_perturbation)
    al = jnp.zeros_like(p)
    alt = jnp.ones_like(p)
    dry_cqu, dry_cqv = _ones_cq(state)
    moist_cqu, moist_cqv = moisture_coupling_factors(moist_state)

    dry_du, _, _, _ = horizontal_pressure_gradient(
        state,
        _base(state),
        metrics,
        p,
        al,
        alt,
        dry_cqu,
        dry_cqv,
        non_hydrostatic=False,
    )
    moist_du, _, _, _ = horizontal_pressure_gradient(
        moist_state,
        _base(moist_state),
        scaled_metrics,
        p,
        al,
        alt,
        moist_cqu,
        moist_cqv,
        non_hydrostatic=False,
    )

    interior_ratio = moist_du[:, :, 1:-1] / dry_du[:, :, 1:-1]
    assert jnp.allclose(interior_ratio, 2.0 / 1.5)


def test_pgf_jaxpr_has_no_host_callbacks():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _state(grid)
    base = _base(state)
    p = jnp.zeros_like(state.p_perturbation)
    al = jnp.zeros_like(p)
    alt = jnp.ones_like(p)
    cqu, cqv = _ones_cq(state)

    jaxpr = str(
        jax.make_jaxpr(
            lambda s, b, m, p_, al_, alt_, cqu_, cqv_: horizontal_pressure_gradient(
                s,
                b,
                m,
                p_,
                al_,
                alt_,
                cqu_,
                cqv_,
                non_hydrostatic=True,
            )
        )(
            state,
            base,
            metrics,
            p,
            al,
            alt,
            cqu,
            cqv,
        )
    ).lower()

    assert "host_callback" not in jaxpr
    assert "io_callback" not in jaxpr
    assert "pure_callback" not in jaxpr
