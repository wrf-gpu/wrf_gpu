"""Tier-3 short-run convergence for a smooth periodic advection oracle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.dynamics.step import run
from gpuwrf.profiling.transfer_audit import block_until_ready
from gpuwrf.validation.tier2 import make_ideal_grid


config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts" / "m4" / "tier3_convergence.json"


def _run_level(dx: float, dt: float, n_steps: int, velocity: float = 20.0) -> dict[str, float]:
    """Runs one public-dycore grid level and returns its analytic 2-norm error."""

    domain_m = 128_000.0
    nx = int(round(domain_m / dx))
    x = jnp.arange(nx, dtype=jnp.float64) * dx
    perturbation = jnp.sin(2.0 * jnp.pi * x / domain_m)
    grid = make_ideal_grid(4, 4, nx, dx_m=dx, dy_m=dx, top_m=480.0)
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    theta = 300.0 + perturbation[None, None, :] + jnp.zeros((grid.nz, grid.ny, 1), dtype=jnp.float64)
    state = state.replace(
        u=jnp.ones_like(state.u) * velocity,
        v=jnp.zeros_like(state.v),
        w=jnp.zeros_like(state.w),
        theta=theta,
        qv=jnp.ones_like(state.qv) * 1.0e-3,
        p=jnp.zeros_like(state.p),
        ph=jnp.zeros_like(state.ph),
        mu=jnp.ones_like(state.mu),
    )
    final = run(state, tendencies, grid, dt, int(n_steps), n_acoustic=1, debug=False)
    block_until_ready(final)
    final_t = float(dt) * float(n_steps)
    exact = jnp.sin(2.0 * jnp.pi * (x - velocity * final_t) / domain_m)
    candidate = final.theta[0, 0, :] - 300.0
    err = jnp.sqrt(jnp.mean((candidate - exact) ** 2))
    return {"dx_m": float(dx), "dt_s": float(dt), "n_steps": int(n_steps), "l2_error": float(err)}


def convergence_record() -> dict[str, Any]:
    """Computes observed order from the three mandated resolution levels."""

    levels = [_run_level(2000.0, 2.0, 50), _run_level(1000.0, 1.0, 100), _run_level(500.0, 0.5, 200)]
    e0, e1, e2 = (level["l2_error"] for level in levels)
    order01 = float(np.log2(e0 / e1))
    order12 = float(np.log2(e1 / e2))
    observed = 0.5 * (order01 + order12)
    expected = 3.0
    return {
        "case": "1d-periodic-sine-linear-advection-public-dycore-run",
        "errors_per_level": levels,
        "observed_order": observed,
        "observed_order_pairs": [order01, order12],
        "expected_order": expected,
        "pass": bool(observed >= expected - 0.5),
    }


def run_tier3(out: Path = ARTIFACT) -> dict[str, Any]:
    """Writes the tier-3 convergence proof JSON for the contract command."""

    record = convergence_record()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
