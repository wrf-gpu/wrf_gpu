from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.dynamics.acoustic_wrf import vertical_acoustic_update
from gpuwrf.dynamics.metrics import flat_metrics_for_grid
from gpuwrf.dynamics.vertical_implicit_solver import build_epssm_column_coefficients


ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / ".agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_theta_explosion.json"


def _grid(nz: int = 44) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    return GridSpec(
        Projection("lambert", 0.0, 0.0, 500.0, 500.0, 1, 1),
        TerrainProvenance("analytic://m6b-coftz-theta-fix", "m6b-coftz", (1, 1), "m", "flat-column", 0.0, True),
        VerticalCoord("hybrid_eta", nz, 5000.0, eta),
        BCMetadata("ideal", ("w", "ph", "theta"), 0, "linear", True),
        eta,
        jnp.zeros((1, 1), dtype=jnp.float64),
    )


def _state_from_bad_cell_proof() -> tuple[State, BaseState, GridSpec, int]:
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    cell = proof["field_snapshots"]["cell"]
    before = proof["field_snapshots"]["step_n_minus_1"]
    nz = 44
    grid = _grid(nz)
    arrays = {field: jnp.zeros(shape, dtype=jnp.float64) for field, shape in _state_field_shapes(grid).items()}

    theta_base = jnp.ones((nz, 1, 1), dtype=jnp.float64) * float(before["theta"])
    z_faces = jnp.linspace(0.0, 220000.0, nz + 1, dtype=jnp.float64)
    phb = 9.80665 * z_faces[:, None, None]
    pb = jnp.ones((nz, 1, 1), dtype=jnp.float64) * 90_000.0
    mub = jnp.ones((1, 1), dtype=jnp.float64) * float(before["mu"])
    w = jnp.zeros((nz + 1, 1, 1), dtype=jnp.float64).at[int(cell["k"]) + 1, 0, 0].set(float(before["w"]))

    arrays.update(
        theta=theta_base,
        w=w,
        p=pb,
        p_total=pb,
        p_perturbation=jnp.zeros_like(pb),
        ph=phb,
        ph_total=phb,
        ph_perturbation=jnp.zeros_like(phb),
        mu=mub,
        mu_total=mub,
        mu_perturbation=jnp.zeros_like(mub),
    )
    return State(**arrays), BaseState(pb=pb, phb=phb, mub=mub, t0=theta_base, theta_base=theta_base), grid, int(cell["k"])


def test_coftz_can_use_stable_theta_reference_when_state_theta_spikes():
    state, base, _grid_spec, k = _state_from_bad_cell_proof()
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    runaway_theta = float(proof["field_snapshots"]["step_n"]["theta"])
    dz = jnp.ones_like(state.theta) * 5000.0
    spiked_theta = state.theta.at[k, 0, 0].set(runaway_theta)

    _cofrz, _cofwr, _cofwz, coftz, _cofwt, _rdzw, _a, _b, _c = build_epssm_column_coefficients(
        spiked_theta,
        dz,
        dt=1.0,
        epssm=0.1,
        theta_coefficient=base.theta_base,
    )

    assert float(coftz[k + 1, 0, 0]) < 250.0


def test_bad_cell_pre_breach_vertical_update_keeps_substep_dtheta_below_one_kelvin():
    state, base, grid, k = _state_from_bad_cell_proof()

    next_state = vertical_acoustic_update(
        state,
        base,
        flat_metrics_for_grid(grid),
        dt=1.0,
        epssm=0.1,
        pressure_scale=-1.0,
        top_lid=True,
    )

    dtheta = abs(float(next_state.theta[k, 0, 0] - state.theta[k, 0, 0]))
    assert dtheta < 1.0
