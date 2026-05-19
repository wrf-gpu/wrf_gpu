"""Tier-3 short-run convergence for a smooth periodic advection oracle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.advection import ddx4_centered


config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts" / "m4" / "tier3_convergence.json"


def _rk3_step(phi, velocity: float, dx: float, dt: float):
    """Advances the one-dimensional linear advection equation by one RK3 step."""

    def rhs(value):
        """Keeps the advection RHS local to the manufactured convergence test."""

        return -velocity * ddx4_centered(value, dx, axis=0)

    k1 = rhs(phi)
    y1 = phi + dt * k1
    k2 = rhs(y1)
    y2 = 0.75 * phi + 0.25 * (y1 + dt * k2)
    k3 = rhs(y2)
    return (phi + 2.0 * (y2 + dt * k3)) / 3.0


def _run_level(dx: float, dt: float, n_steps: int, velocity: float = 20.0) -> dict[str, float]:
    """Runs one grid level and returns its analytic 2-norm error."""

    domain_m = 128_000.0
    nx = int(round(domain_m / dx))
    x = jnp.arange(nx, dtype=jnp.float64) * dx
    center0 = 0.35 * domain_m
    width = 0.08 * domain_m
    phi = jnp.exp(-((x - center0) / width) ** 2)
    for _ in range(int(n_steps)):
        phi = _rk3_step(phi, velocity, dx, dt)
    final_t = float(dt) * float(n_steps)
    shifted = jnp.mod(x - velocity * final_t, domain_m)
    exact = jnp.exp(-((shifted - center0) / width) ** 2)
    err = jnp.sqrt(jnp.mean((phi - exact) ** 2))
    return {"dx_m": float(dx), "dt_s": float(dt), "n_steps": int(n_steps), "l2_error": float(err)}


def convergence_record() -> dict[str, Any]:
    """Computes observed order from the three mandated resolution levels."""

    levels = [_run_level(2000.0, 20.0, 100), _run_level(1000.0, 10.0, 200), _run_level(500.0, 5.0, 400)]
    e0, e1, e2 = (level["l2_error"] for level in levels)
    order01 = float(np.log2(e0 / e1))
    order12 = float(np.log2(e1 / e2))
    observed = 0.5 * (order01 + order12)
    expected = 3.0
    return {
        "case": "1d-periodic-smooth-bump-linear-advection-rk3-centered-reference",
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
