from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import jax.numpy as jnp

from gpuwrf.dynamics.acoustic_wrf import (
    AcousticConfig,
    MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
    _mpas_recurrence_vertical_update,
    acoustic_substep_carry,
    diagnose_pressure_al_alt,
    initialize_acoustic_carry,
)
from gpuwrf.dynamics.metrics import flat_metrics_for_grid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostic_warm_bubble_vs_slice import build_warm_bubble_3d


DT_ACOUSTIC_S = 0.25


def _warm_bubble_inputs():
    grid, state, base, _theta_base, _z_mass = build_warm_bubble_3d()
    return state, base, flat_metrics_for_grid(grid)


def _max_abs(array) -> float:
    return float(np.max(np.abs(np.asarray(array, dtype=np.float64))))


def test_nonhydrostatic_carry_preserves_recurrence_pressure_perturbation():
    state, base, metrics = _warm_bubble_inputs()

    recurrence_state = _mpas_recurrence_vertical_update(
        state,
        base,
        metrics,
        dt=DT_ACOUSTIC_S,
        epssm=0.1,
        top_lid=True,
        buoyancy_scale=MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
    )
    diagnostic_pressure, _diagnostic_al, _diagnostic_alt = diagnose_pressure_al_alt(
        recurrence_state,
        base,
        metrics,
    )
    config = AcousticConfig(
        n_substeps=8,
        dx_m=400.0,
        dy_m=400.0,
        non_hydrostatic=True,
        top_lid=True,
        mu_continuity=True,
        epssm=0.1,
    )
    carry = initialize_acoustic_carry(state, state.p_perturbation, metrics, base, config)

    updated = acoustic_substep_carry(carry, metrics, config, DT_ACOUSTIC_S, base)

    assert _max_abs(recurrence_state.p_perturbation) > 1.0
    assert _max_abs(diagnostic_pressure) < 1.0e-6
    np.testing.assert_allclose(
        np.asarray(updated.state.p_perturbation),
        np.asarray(recurrence_state.p_perturbation),
        rtol=0.0,
        atol=1.0e-9,
    )


def test_hydrostatic_carry_keeps_diagnostic_pressure_replacement():
    state, base, metrics = _warm_bubble_inputs()
    config = AcousticConfig(
        n_substeps=8,
        dx_m=400.0,
        dy_m=400.0,
        non_hydrostatic=False,
        top_lid=True,
        mu_continuity=True,
        epssm=0.1,
    )
    carry = initialize_acoustic_carry(state, state.p_perturbation, metrics, base, config)

    updated = acoustic_substep_carry(carry, metrics, config, DT_ACOUSTIC_S, base)
    diagnostic_pressure, _diagnostic_al, diagnostic_alt = diagnose_pressure_al_alt(
        updated.state,
        base,
        metrics,
    )

    np.testing.assert_allclose(
        np.asarray(updated.state.p_perturbation),
        np.asarray(diagnostic_pressure),
        rtol=0.0,
        atol=1.0e-9,
    )
    assert bool(jnp.all(jnp.isfinite(diagnostic_alt)))
