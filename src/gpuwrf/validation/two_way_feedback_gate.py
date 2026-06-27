"""v0.22 two-way nesting feedback validation gate.

The gate is intentionally CPU-runnable: it drives the real domain-tree feedback
callback over a minimal idealized d01->d02 nest and checks the invariants that
matter for wiring correctness before a longer GPU forecast smoke is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.coupling.boundary_feedback import (
    apply_feedback,
    apply_state_feedback,
    feedback_mask,
    feedback_overlap_conservation,
    sm121_smooth,
)
from gpuwrf.runtime.domain_tree import DomainEdge, run_domain_tree_callbacks


@dataclass(frozen=True)
class _MiniGrid:
    ny: int
    nx: int


class _MiniState:
    """Small duck-typed state exposing the leaves feedback reads/writes."""

    _FIELDS = (
        "u",
        "v",
        "w",
        "theta",
        "qv",
        "p_perturbation",
        "p_total",
        "p",
        "ph_perturbation",
        "ph_total",
        "ph",
        "mu_perturbation",
        "mu_total",
        "mu",
        "qke",
    )

    def __init__(self, **kwargs: Any) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)

    def replace(self, _cast: bool = True, **updates: Any) -> "_MiniState":
        del _cast
        values = {name: getattr(self, name) for name in self._FIELDS}
        values.update(updates)
        return _MiniState(**values)


@dataclass(frozen=True)
class _Carry:
    state: _MiniState


def _mini_state(ny: int, nx: int, *, nz: int, fill: float, base: float) -> _MiniState:
    mass3 = jnp.full((nz, ny, nx), fill, dtype=jnp.float64)
    mass2 = jnp.full((ny, nx), fill, dtype=jnp.float64)
    u = jnp.full((nz, ny, nx + 1), fill, dtype=jnp.float64)
    v = jnp.full((nz, ny + 1, nx), fill, dtype=jnp.float64)
    p_base = jnp.full_like(mass3, base)
    ph_base = jnp.full_like(mass3, base + 1000.0)
    mu_base = jnp.full_like(mass2, base + 2000.0)
    return _MiniState(
        u=u,
        v=v,
        w=mass3,
        theta=mass3,
        qv=mass3,
        p_perturbation=mass3,
        p_total=p_base + mass3,
        p=p_base + mass3,
        ph_perturbation=mass3,
        ph_total=ph_base + mass3,
        ph=ph_base + mass3,
        mu_perturbation=mass2,
        mu_total=mu_base + mass2,
        mu=mu_base + mass2,
        qke=mass3,
    )


def _with_child_tendency(state: _MiniState, delta: float) -> _MiniState:
    """Advance the child by a uniform deterministic signal.

    A uniform signal makes the WRF ``sm121`` pass conservative for this gate while
    still proving that the post-subcycle child value reaches the parent.
    """

    inc3 = jnp.asarray(delta, dtype=jnp.float64)
    inc2 = jnp.asarray(delta, dtype=jnp.float64)
    p_base = state.p_total - state.p_perturbation
    ph_base = state.ph_total - state.ph_perturbation
    mu_base = state.mu_total - state.mu_perturbation
    p_pert = state.p_perturbation + inc3
    ph_pert = state.ph_perturbation + inc3
    mu_pert = state.mu_perturbation + inc2
    return state.replace(
        _cast=False,
        u=state.u + inc3,
        v=state.v + inc3,
        w=state.w + inc3,
        theta=state.theta + inc3,
        qv=state.qv + inc3,
        qke=state.qke + inc3,
        p_perturbation=p_pert,
        p_total=p_base + p_pert,
        p=p_base + p_pert,
        ph_perturbation=ph_pert,
        ph_total=ph_base + ph_pert,
        ph=ph_base + ph_pert,
        mu_perturbation=mu_pert,
        mu_total=mu_base + mu_pert,
        mu=mu_base + mu_pert,
    )


def _all_finite(state: _MiniState) -> bool:
    return all(bool(np.isfinite(np.asarray(getattr(state, name))).all()) for name in state._FIELDS)


def _feedback_ring(mask: np.ndarray) -> np.ndarray:
    north = np.zeros_like(mask, dtype=bool)
    south = np.zeros_like(mask, dtype=bool)
    west = np.zeros_like(mask, dtype=bool)
    east = np.zeros_like(mask, dtype=bool)
    north[:-1, :] = mask[1:, :]
    south[1:, :] = mask[:-1, :]
    west[:, 1:] = mask[:, :-1]
    east[:, :-1] = mask[:, 1:]
    interior = mask & north & south & west & east
    return mask & ~interior


def run_gate(*, output: str | Path | None = None) -> dict[str, Any]:
    """Run the minimal d01->d02 feedback validation and optionally write JSON."""

    from gpuwrf.coupling.boundary_feedback import build_state_feedback_weights

    ratio = 3
    i_parent_start = 2
    j_parent_start = 2
    parent_grid = _MiniGrid(ny=16, nx=16)
    child_grid = _MiniGrid(ny=24, nx=24)
    nz = 3
    weights = build_state_feedback_weights(
        parent_grid_ratio=ratio,
        i_parent_start=i_parent_start,
        j_parent_start=j_parent_start,
        parent_grid=parent_grid,
        child_grid=child_grid,
        spec_zone=1,
        smooth_option=1,
    )
    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"),
        (
            DomainNest(
                "d01",
                "d02",
                ratio,
                i_parent_start,
                j_parent_start,
                feedback=True,
            ),
        ),
        max_dom=2,
    )
    spec = hierarchy.nests[0]
    runtime_edge = DomainEdge(spec=spec, weights=None, feedback_weights=weights)  # type: ignore[arg-type]

    initial_parent = _mini_state(parent_grid.ny, parent_grid.nx, nz=nz, fill=1.0, base=100.0)
    initial_child = _mini_state(child_grid.ny, child_grid.nx, nz=nz, fill=11.0, base=900.0)

    def advance(name: str, carry: _Carry, start_step: int, n_steps: int) -> _Carry:
        if name == "d02":
            return _Carry(_with_child_tendency(carry.state, 0.5 * float(n_steps)))
        return carry

    def force(edge: DomainEdge, parent: _Carry, child: _Carry) -> _Carry:
        del edge, parent
        return child

    def feedback(edge: DomainEdge, parent: _Carry, child: _Carry) -> _Carry:
        fed = apply_state_feedback(parent.state, child.state, edge.feedback_weights, feedback=True)
        return _Carry(fed)

    result = run_domain_tree_callbacks(
        hierarchy,
        {"d01": _Carry(initial_parent), "d02": _Carry(initial_child)},
        root_steps=2,
        advance=advance,
        force=force,
        feedback=feedback,
        feedback_enabled=True,
        edge_lookup=lambda _spec: runtime_edge,
        block_between=False,
    )
    parent = result.states["d01"]
    child = result.states["d02"]

    mask = np.asarray(feedback_mask(weights.mass), dtype=bool)
    ring = _feedback_ring(mask)
    theta_before = np.asarray(initial_parent.theta)
    theta_parent = np.asarray(parent.theta)
    theta_child = np.asarray(child.theta)
    fed_only = apply_feedback(initial_parent.theta, child.theta, weights.mass, feedback=True)
    expected = sm121_smooth(fed_only, weights.mass_smooth)
    theta_expected = np.asarray(expected)

    cons = feedback_overlap_conservation(child.theta, weights.mass, leaf="theta")
    outside = ~mask
    outside_max_abs_change = float(np.max(np.abs(theta_parent[:, outside] - theta_before[:, outside])))
    overlap_max_abs_error = float(np.max(np.abs(theta_parent[:, mask] - theta_expected[:, mask])))
    ring_mean_delta = float(np.mean(np.abs(theta_parent[:, ring] - theta_before[:, ring])))
    ring_expected_mean = float(np.mean(theta_expected[:, ring]))
    ring_parent_mean = float(np.mean(theta_parent[:, ring]))

    events = tuple(tuple(item) for item in result.events)
    feedback_event_count = sum(1 for event in events if event and event[0] == "feedback")
    force_event_count = sum(1 for event in events if event and event[0] == "force")
    passed = (
        feedback_event_count == 2
        and force_event_count == 2
        and result.own_steps == {"d01": 2, "d02": 6}
        and _all_finite(parent)
        and _all_finite(child)
        and bool(cons.conserved)
        and cons.rel_residual <= 1.0e-12
        and overlap_max_abs_error <= 1.0e-12
        and outside_max_abs_change <= 1.0e-12
        and ring_mean_delta > 1.0
    )
    payload: dict[str, Any] = {
        "schema": "gpuwrf.v022.two_way_feedback_gate",
        "schema_version": 1,
        "verdict": "PASS" if passed else "FAIL",
        "fixture": {
            "kind": "idealized_minimal_two_domain_nest",
            "parent_shape": [nz, parent_grid.ny, parent_grid.nx],
            "child_shape": [nz, child_grid.ny, child_grid.nx],
            "parent_grid_ratio": ratio,
            "i_parent_start": i_parent_start,
            "j_parent_start": j_parent_start,
            "root_steps": 2,
            "native_subcycle_steps": int(result.own_steps["d02"]),
            "smooth_option": 1,
        },
        "wrf_reference": {
            "root": "<DATA_ROOT>/src/wrf_pristine",
            "driver": "WRF/share/mediation_integrate.F::med_nest_feedback",
            "feedback": "WRF/share/interp_fcn.F::copy_fcn",
            "smoothing": "WRF/share/interp_fcn.F::sm121",
        },
        "scheduler": {
            "own_steps": dict(result.own_steps),
            "event_counts": {
                "advance": sum(1 for event in events if event and event[0] == "advance"),
                "force": force_event_count,
                "feedback": feedback_event_count,
            },
            "events": [list(event) for event in events],
        },
        "finite": {
            "parent": _all_finite(parent),
            "child": _all_finite(child),
        },
        "conservation": {
            "leaf": cons.leaf,
            "stagger": cons.stagger,
            "n_parent_cells": cons.n_cells,
            "child_overlap_integral": cons.child_overlap_integral,
            "parent_overlap_integral": cons.parent_overlap_integral,
            "abs_residual": cons.abs_residual,
            "rel_residual": cons.rel_residual,
            "conserved": cons.conserved,
            "rel_tol": 1.0e-12,
        },
        "parent_reflects_child": {
            "feedback_overlap_parent_cell_count": int(mask.sum()),
            "feedback_boundary_ring_cell_count": int(ring.sum()),
            "overlap_max_abs_error_vs_copyfcn_then_sm121": overlap_max_abs_error,
            "outside_overlap_max_abs_change": outside_max_abs_change,
            "boundary_ring_parent_mean": ring_parent_mean,
            "boundary_ring_expected_child_feedback_mean": ring_expected_mean,
            "boundary_ring_mean_abs_delta_from_initial_parent": ring_mean_delta,
            "passed": ring_mean_delta > 1.0 and overlap_max_abs_error <= 1.0e-12,
        },
        "transfer_audit": {
            "gpu_used": False,
            "host_device_transfer_inside_timestep_loop": False,
            "note": "CPU JAX/dummy-state gate; production GPU smoke must use scripts/with_gpu_lock.sh.",
        },
        "argv": sys.argv,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
