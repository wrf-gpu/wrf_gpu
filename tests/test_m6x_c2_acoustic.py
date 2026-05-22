from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.dynamics.acoustic_wrf import (
    AcousticConfig,
    acoustic_substep,
    diagnose_pressure_al_alt,
    initialize_acoustic_carry,
    moisture_coupling_factors,
    mu_continuity_tendency,
    run_acoustic_scan_carry,
)
from gpuwrf.dynamics.damping import SmdivConfig
from gpuwrf.dynamics.metrics import flat_metrics_for_grid


def _rest_state_and_base(grid: GridSpec) -> tuple[State, BaseState]:
    arrays = {field: jnp.zeros(shape, dtype=jnp.float64) for field, shape in _state_field_shapes(grid).items()}
    theta = jnp.ones_like(arrays["theta"]) * 300.0
    pb = jnp.ones_like(arrays["p"]) * 90000.0
    phb = jnp.zeros_like(arrays["ph"])
    mub = jnp.ones_like(arrays["mu"]) * 90000.0
    arrays["theta"] = theta
    arrays["p"] = pb
    arrays["p_total"] = pb
    arrays["p_perturbation"] = jnp.zeros_like(pb)
    arrays["ph"] = phb
    arrays["ph_total"] = phb
    arrays["ph_perturbation"] = jnp.zeros_like(phb)
    arrays["mu"] = mub
    arrays["mu_total"] = mub
    arrays["mu_perturbation"] = jnp.zeros_like(mub)
    state = State(**arrays)
    base = BaseState(pb=pb, phb=phb, mub=mub, t0=theta, theta_base=theta)
    return state, base


def test_diagnostic_pressure_al_alt_matches_base_rest_state():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state, base = _rest_state_and_base(grid)

    pressure, al, alt = diagnose_pressure_al_alt(state, base, metrics)

    assert float(jnp.max(jnp.abs(pressure))) < 1.0e-8
    assert float(jnp.max(jnp.abs(al))) < 1.0e-12
    assert jnp.all(jnp.isfinite(alt))
    assert float(jnp.min(alt)) > 0.0


def test_acoustic_carry_includes_al_alt_and_cq_intermediates():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state, base = _rest_state_and_base(grid)

    carry = initialize_acoustic_carry(state, state.p_perturbation, metrics, base, AcousticConfig())

    assert carry.al.shape == state.p_perturbation.shape
    assert carry.alt.shape == state.p_perturbation.shape
    assert carry.cqu.shape == state.u.shape
    assert carry.cqv.shape == state.v.shape
    assert jnp.allclose(carry.cqu, 1.0)
    assert jnp.allclose(carry.cqv, 1.0)


def test_smdiv_pressure_memory_preserves_previous_diagnostic_pressure():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state, _ = _rest_state_and_base(grid)
    legacy_state = state.replace(p_total=jnp.ones_like(state.p_total) * 10.0, p_perturbation=jnp.ones_like(state.p_perturbation) * 10.0)
    previous = legacy_state.p_perturbation - 1.0

    next_state, next_previous = acoustic_substep(
        legacy_state,
        previous,
        metrics,
        AcousticConfig(smdiv=SmdivConfig(enabled=True, coefficient=0.1), mu_continuity=False),
        dt=1.0,
        base_state=None,
    )

    assert jnp.allclose(next_previous, legacy_state.p_perturbation)
    assert jnp.allclose(next_state.p_perturbation, legacy_state.p_perturbation + 0.1)


def test_mu_continuity_tendency_is_nonzero_for_divergent_u_flux():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state, base = _rest_state_and_base(grid)
    x_face = jnp.arange(grid.nx + 1, dtype=jnp.float64)[None, None, :]
    state = state.replace(u=jnp.broadcast_to(x_face, state.u.shape))

    dmu_dt = mu_continuity_tendency(state, base, metrics, dx_m=1.0, dy_m=1.0)

    assert float(jnp.max(jnp.abs(dmu_dt))) > 0.0
    assert jnp.all(jnp.isfinite(dmu_dt))


def test_acoustic_scan_updates_mu_inside_substep_loop():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state, base = _rest_state_and_base(grid)
    x_face = jnp.arange(grid.nx + 1, dtype=jnp.float64)[None, None, :]
    state = state.replace(u=jnp.broadcast_to(x_face, state.u.shape))

    carry = run_acoustic_scan_carry(
        state,
        state.p_perturbation,
        metrics,
        AcousticConfig(n_substeps=2, dx_m=1.0, dy_m=1.0, non_hydrostatic=False),
        2.0,
        base,
    )

    assert float(jnp.max(jnp.abs(carry.state.mu_perturbation))) > 0.0
    assert jnp.allclose(carry.state.mu_total, base.mub + carry.state.mu_perturbation)


def test_diagnostic_pressure_is_recomputed_after_mu_changes():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state, base = _rest_state_and_base(grid)
    x_face = jnp.arange(grid.nx + 1, dtype=jnp.float64)[None, None, :]
    state = state.replace(u=jnp.broadcast_to(x_face, state.u.shape))

    carry = run_acoustic_scan_carry(
        state,
        state.p_perturbation,
        metrics,
        AcousticConfig(n_substeps=2, dx_m=1.0, dy_m=1.0, non_hydrostatic=False),
        2.0,
        base,
    )

    assert float(jnp.max(jnp.abs(carry.state.p_perturbation))) > 0.0
    assert jnp.all(jnp.isfinite(carry.alt))


def test_moisture_coupling_factors_are_scan_carried():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state, base = _rest_state_and_base(grid)
    state = state.replace(qv=jnp.ones_like(state.qv) * 0.5)

    cqu, cqv = moisture_coupling_factors(state)
    carry = initialize_acoustic_carry(state, state.p_perturbation, metrics, base, AcousticConfig())

    assert jnp.allclose(cqu, 1.0 / 1.5)
    assert jnp.allclose(cqv, 1.0 / 1.5)
    assert jnp.allclose(carry.cqu, cqu)
    assert jnp.allclose(carry.cqv, cqv)


def test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state, base = _rest_state_and_base(grid)
    config = AcousticConfig(n_substeps=2, non_hydrostatic=False)

    jaxpr = str(
        jax.make_jaxpr(run_acoustic_scan_carry, static_argnums=(3, 4))(
            state,
            state.p_perturbation,
            metrics,
            config,
            1.0,
            base,
        )
    ).lower()

    assert "scan" in jaxpr
    assert "host_callback" not in jaxpr
    assert "io_callback" not in jaxpr
    assert "pure_callback" not in jaxpr
