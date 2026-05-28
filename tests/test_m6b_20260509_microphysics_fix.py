from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gpuwrf.physics.thompson_column import (
    ThompsonColumnState,
    density_from_pressure_temperature,
    step_thompson_column,
)


R_D_OVER_CP = 287.05 / 1004.0
P0_PA = 100000.0


def _single_cell_state(*, theta: float, p: float, qv: float, qc: float = 0.0) -> ThompsonColumnState:
    pressure = jnp.asarray([[p]], dtype=jnp.float64)
    exner = (jnp.maximum(pressure, 1.0) / P0_PA) ** R_D_OVER_CP
    temperature = jnp.asarray([[theta]], dtype=jnp.float64) * exner
    vapor = jnp.asarray([[qv]], dtype=jnp.float64)
    cloud = jnp.asarray([[qc]], dtype=jnp.float64)
    zeros = jnp.zeros_like(vapor)
    rho = density_from_pressure_temperature(pressure, temperature, vapor)
    return ThompsonColumnState(vapor, cloud, zeros, zeros, zeros, zeros, zeros, zeros, temperature, pressure, rho)


def _theta_from_output(state: ThompsonColumnState) -> np.ndarray:
    exner = (np.maximum(np.asarray(state.p), 1.0) / P0_PA) ** R_D_OVER_CP
    return np.asarray(state.T) / np.maximum(exner, 1.0e-12)


def test_20260509_bad_cell_invalid_pressure_does_not_create_cloud_feedback():
    state = _single_cell_state(
        theta=348.63739013671875,
        p=-34582.75401226425,
        qv=0.00011934670328628272,
        qc=0.0,
    )

    out = step_thompson_column(state, 10.0, debug=False)

    dqc = float(np.asarray(out.qc - state.qc)[0, 0])
    dtheta = float((_theta_from_output(out) - _theta_from_output(state))[0, 0])
    assert np.isfinite(dqc)
    assert np.isfinite(dtheta)
    assert dqc < 0.005
    assert abs(dtheta) < 5.0


def test_realistic_supersaturated_cell_keeps_bounded_condensation_active():
    state = _single_cell_state(theta=298.5550620167698, p=85000.0, qv=0.012, qc=1.0e-4)

    out = step_thompson_column(state, 10.0, debug=False)

    dqc = float(np.asarray(out.qc - state.qc)[0, 0])
    dtheta = float((_theta_from_output(out) - _theta_from_output(state))[0, 0])
    assert 0.0 < dqc < 0.005
    assert 0.0 < dtheta < 5.0
