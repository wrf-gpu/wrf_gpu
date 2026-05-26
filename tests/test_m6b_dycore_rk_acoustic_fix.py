from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from gpuwrf.contracts.state import Tendencies
from gpuwrf.runtime.operational_mode import _m6b_acoustic_tendencies


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-26-m6b-dycore-rk-acoustic-fix"


def _zero_tendencies(v_shape: tuple[int, ...]) -> Tendencies:
    mass_shape = (v_shape[0], v_shape[1] - 1, v_shape[2])
    u_shape = (v_shape[0], v_shape[1] - 1, v_shape[2] + 1)
    w_shape = (v_shape[0] + 1, v_shape[1] - 1, v_shape[2])
    surface_shape = (v_shape[1] - 1, v_shape[2])
    return Tendencies(
        u=jnp.zeros(u_shape, dtype=jnp.float64),
        v=jnp.zeros(v_shape, dtype=jnp.float64),
        w=jnp.zeros(w_shape, dtype=jnp.float64),
        theta=jnp.zeros(mass_shape, dtype=jnp.float64),
        qv=jnp.zeros(mass_shape, dtype=jnp.float64),
        p=jnp.zeros(mass_shape, dtype=jnp.float64),
        ph=jnp.zeros(w_shape, dtype=jnp.float64),
        mu=jnp.zeros(surface_shape, dtype=jnp.float64),
    )


def test_step46_v_runaway_tendency_is_suppressed():
    proof = json.loads((SPRINT / "baseline" / "proof_step46_violation.json").read_text(encoding="utf-8"))
    budget = json.loads((SPRINT / "baseline" / "proof_operator_decomposition.json").read_text(encoding="utf-8"))
    loc = tuple(proof["field_snapshots"]["step46"]["v"]["index"])
    v_shape = tuple(proof["field_snapshots"]["step46"]["v"]["shape"])
    dt_s = float(proof["namelist"]["dt_s"])
    runaway_dv_dt = float(budget["terms_per_second"]["dycore_rk_acoustic"][0])

    reduced = _zero_tendencies(v_shape).replace(v=jnp.ones(v_shape, dtype=jnp.float64) * runaway_dv_dt)
    base = _zero_tendencies(v_shape)
    fixed = _m6b_acoustic_tendencies(reduced, base)

    k, j, i = loc
    dv = float(fixed.v[k, j, i] * dt_s)
    assert dv < 5.0
