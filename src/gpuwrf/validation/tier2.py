"""Tier-2 invariant checks for the reduced M4 dycore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.dynamics.step import run
from gpuwrf.profiling.transfer_audit import block_until_ready


config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts" / "m4" / "tier2_invariants.json"


def make_ideal_grid(nz: int = 40, ny: int = 80, nx: int = 80, dx_m: float = 250.0, dy_m: float = 250.0) -> GridSpec:
    """Constructs the M4 idealized flat grid reused by scripts and tests."""

    projection = Projection("lambert", 0.0, 0.0, float(dx_m), float(dy_m), int(nx), int(ny))
    terrain = TerrainProvenance(
        source_path="analytic://m4-flat",
        sha256="analytic-m4-flat",
        shape=(int(ny), int(nx)),
        units="m",
        projection_transform="identity",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta_levels = jnp.linspace(1.0, 0.0, int(nz) + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", int(nz), 4000.0, eta_levels)
    bc = BCMetadata("ideal", ("u", "v", "theta", "qv", "p"), 0, "linear", False)
    terrain_height = jnp.zeros((int(ny), int(nx)), dtype=jnp.float64)
    return GridSpec(projection, terrain, vertical, bc, eta_levels, terrain_height)


def density_current_state(grid: GridSpec) -> tuple[State, Tendencies]:
    """Creates the M4 still-air cold-blob state used for invariant validation."""

    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    z = jnp.arange(grid.nz, dtype=jnp.float64)[:, None, None]
    y = jnp.arange(grid.ny, dtype=jnp.float64)[None, :, None]
    x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
    x0 = 0.5 * float(grid.nx - 1)
    z0 = 0.33 * float(grid.nz - 1)
    radius_x = 0.20 * float(grid.nx)
    radius_z = 0.18 * float(grid.nz)
    cold = ((x - x0) / radius_x) ** 2 + ((z - z0) / radius_z) ** 2 <= 1.0
    theta = jnp.where(cold, 285.0, 300.0) + y * 0.0
    qv = theta * 0.0 + 1.0e-3
    p = theta * 0.0 + 1.0
    mu = jnp.ones((grid.ny, grid.nx), dtype=jnp.float64)
    return state.replace(theta=theta, qv=qv, p=p, mu=mu), tendencies


def invariant_record(grid: GridSpec, n_steps: int = 100, dt: float = 2.0, n_acoustic: int = 4) -> dict[str, Any]:
    """Runs the dycore and computes the required tier-2 invariant scalars."""

    state, tendencies = density_current_state(grid)
    mass0 = jnp.sum(state.mu)
    final = run(state, tendencies, grid, dt, int(n_steps), n_acoustic=int(n_acoustic), debug=False)
    block_until_ready(final)
    mass1 = jnp.sum(final.mu)
    mass_residual = jnp.abs(mass1 - mass0) / mass0
    qv_violations = jnp.sum(final.qv < 0.0)
    leaves = jax.tree_util.tree_leaves(final)
    nan_inf = sum(int(jnp.sum(~jnp.isfinite(leaf))) for leaf in leaves)
    record = {
        "case": "m4-still-air-density-current-cold-blob",
        "iterations": int(n_steps),
        "dt_s": float(dt),
        "mass_residual_relative": float(mass_residual),
        "qv_positivity_violations": int(qv_violations),
        "nan_inf_violations": int(nan_inf),
        "pass": bool(float(mass_residual) <= 1.0e-10 and int(qv_violations) == 0 and int(nan_inf) == 0),
    }
    return record


def run_tier2(out: Path = ARTIFACT) -> dict[str, Any]:
    """Writes the tier-2 invariant proof JSON for the contract command."""

    record = invariant_record(make_ideal_grid())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
