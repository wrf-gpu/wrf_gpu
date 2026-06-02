"""Nest scheduler / subcycling cadence (WRF ``module_integrate`` faithful).

Pure-HOST orchestration of the parent->child subcycling cadence and the
``med_nest_force`` forcedown ordering.  This module defines the cadence and the
recursion STRUCTURE (and a numeric substep-count audit); it does NOT drive the
device step loop -- the runtime hook that wires this into the operational step
is SPEC'd for the manager in :func:`runtime_hook_spec` and in
``proofs/p0_1/FINDINGS.md`` (P0-1a delivers the cadence + construction; the live
device integration is P0-1b).

WRF ground truth (``frame/module_integrate.F:408-435``):

    integrate(grid):
        for each of grid's own steps:
            solve_interface(grid)           ! advance parent ONE parent step
            domain_clockadvance(grid)
            for kid in nests(grid):
                med_nest_force(grid, kid)   ! forcedown: parent -> child boundary
            for kid in nests(grid):
                integrate(kid)              ! RECURSE: ratio child substeps
            (feedback child -> parent is applied at the child stop-subtime)

So the ordering per parent step is strictly: **(1) advance parent one step ->
(2) force ALL children's boundaries from the just-advanced parent -> (3) recurse
each child for ``parent_grid_ratio`` substeps -> (4) optional feedback.**  A child
that is itself a parent recurses; a leaf child runs ``ratio`` substeps.

Boundary cadence (the piece the v0.1.0 hourly replay lacked): each child's
``update_cadence_s`` is its PARENT timestep, and its ``*_bdy`` leaves are the
two-time ``[old_child_ring, new_parent_target]`` package
(:func:`gpuwrf.nesting.boundary_construction.build_child_boundary_package`) built
ONCE per parent step.  The child runs ``ratio`` substeps with local lead
``lead_seconds = step * dt_child`` sweeping ``[dt_child, parent_dt]`` so the
linear time interpolation reproduces WRF ``bdy_* + dtbc*bdy_t*`` over the
subcycle (``interp_fcn.F:2578-2617``).

This module is GPU-free and import-light: it carries only static geometry and a
host callback signature so it can be unit-tested with a pure-Python "advance"
stub (the P0-1a cadence proof) and later bound to the device segment by the
manager (P0-1b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class NestEdge:
    """Static parent->child nest geometry for one edge of the tower.

    ``parent_grid_ratio`` is the WRF spatial+temporal subcycle ratio (3 for our
    9->3 km and 3->1 km nests).  ``i_parent_start`` / ``j_parent_start`` are the
    1-based WRF nest origin offsets in the PARENT grid.
    """

    parent: str
    child: str
    parent_grid_ratio: int
    i_parent_start: int
    j_parent_start: int


@dataclass
class NestTower:
    """A nested tower: the coarse->fine domain order + the per-parent edge list.

    ``order`` lists domains coarse->fine (e.g. ["d01", "d02", "d03"]).
    ``edges[parent]`` is the list of that parent's child edges.
    """

    order: list[str]
    edges: dict[str, list[NestEdge]] = field(default_factory=dict)

    @classmethod
    def from_edges(cls, order: list[str], edges: list[NestEdge]) -> "NestTower":
        by_parent: dict[str, list[NestEdge]] = {}
        for e in edges:
            by_parent.setdefault(e.parent, []).append(e)
        return cls(order=list(order), edges=by_parent)


def expected_substep_counts(
    tower: NestTower, *, root_steps: int, root: str | None = None
) -> dict[str, int]:
    """Total own-steps each domain takes for ``root_steps`` root steps.

    A domain's count = its parent's count x parent_grid_ratio (WRF subcycling).
    For the 9->3->1 km tower with 3:1 ratios and root_steps=N: d01 N, d02 3N,
    d03 9N.  Pure host; the scheduler-recursion correctness gate.
    """

    root = root or tower.order[0]
    counts: dict[str, int] = {root: int(root_steps)}

    def walk(name: str) -> None:
        for e in tower.edges.get(name, []):
            counts[e.child] = counts[name] * int(e.parent_grid_ratio)
            walk(e.child)

    walk(root)
    return counts


def forcedown_event_log(
    tower: NestTower, *, root_steps: int, root: str | None = None
) -> list[tuple[str, str, int]]:
    """The exact ordered event trace WRF ``integrate`` produces (pure host).

    Returns a flat list of events ``(kind, domain, local_step)`` where ``kind`` is
    one of:
      * ``"advance"`` -- the domain takes one of its own steps;
      * ``"force"``   -- ``med_nest_force`` from ``domain``'s just-advanced state
        onto each child (one event per child, ``domain`` is the CHILD here);
      * ``"recurse"`` -- the recursive ``integrate`` entry for a child.

    The trace is the falsifiable cadence object for the P0-1a scheduler proof:
    its length, ordering, and per-domain advance counts must match WRF's
    parent-step -> force-children -> recurse-children structure.
    """

    root = root or tower.order[0]
    events: list[tuple[str, str, int]] = []

    def integrate(name: str, n_steps: int) -> None:
        children = tower.edges.get(name, [])
        for local in range(1, int(n_steps) + 1):
            events.append(("advance", name, local))
            # force every child's boundary from the just-advanced parent
            for e in children:
                events.append(("force", e.child, local))
            # then recurse each child for ratio substeps
            for e in children:
                events.append(("recurse", e.child, local))
                integrate(e.child, int(e.parent_grid_ratio))

    integrate(root, int(root_steps))
    return events


def run_host_tower(
    tower: NestTower,
    states: dict[str, object],
    *,
    root_steps: int,
    advance: Callable[[str, object, int], object],
    force: Callable[[str, object, object], object],
    feedback: Callable[[str, object, object], object] | None = None,
    root: str | None = None,
) -> dict[str, object]:
    """Drive the WRF subcycling recursion with HOST callbacks (no device here).

    ``advance(name, state, local_step) -> new_state`` advances a domain one own
    step.  ``force(child, parent_state, child_state) -> new_child_state`` builds
    the child boundary package from the just-advanced parent.  ``feedback`` (opt)
    is the child->parent two-way feedback applied at each child stop-subtime;
    default-OFF (one-way down-nesting).

    Returns the final per-domain states.  This is the EXACT control structure the
    P0-1b device runtime will follow; here it is exercised with pure-Python stubs
    for the cadence proof, and is the reference the manager binds the device
    segment to.  No field crosses any bus in this function -- it is host
    orchestration of opaque ``state`` objects.
    """

    root = root or tower.order[0]
    out = dict(states)

    def integrate(name: str, n_steps: int) -> None:
        children = tower.edges.get(name, [])
        for local in range(1, int(n_steps) + 1):
            out[name] = advance(name, out[name], local)
            for e in children:
                out[e.child] = force(e.child, out[name], out[e.child])
            for e in children:
                integrate(e.child, int(e.parent_grid_ratio))
                if feedback is not None:
                    out[name] = feedback(name, out[name], out[e.child])

    integrate(root, int(root_steps))
    return out


def runtime_hook_spec() -> str:
    """The single runtime hook the manager wires for P0-1b (text spec).

    Returns the spec string so it is committed alongside the code and unit-testable
    for presence.  The hook is intentionally NOT implemented here (no runtime-step
    edit in P0-1a).
    """

    return (
        "P0-1b runtime hook (manager owns the wiring; do NOT implement in P0-1a):\n"
        "  In runtime/operational_mode, replace the single-domain step loop for a\n"
        "  nested run with scheduler.run_host_tower, binding:\n"
        "    advance(name, carry, local_step) := _advance_chunk(carry, namelist[name],\n"
        "        step_index=local_step, n_steps=1, cadence=radiation_cadence_steps)\n"
        "        -- the child's boundary clock lead_seconds = local_step*dt_child\n"
        "        sweeps [dt_child, parent_dt] so the two-time leaf interpolates as\n"
        "        WRF bdy_*+dtbc*bdy_t*.\n"
        "    force(child, parent_carry, child_carry) :=\n"
        "        boundary_construction.build_child_boundary_package(child_carry.state,\n"
        "        parent_carry.state, edge.weights, bdy_width=spec_bdy_width)\n"
        "        -- ONE device op per parent step, ZERO host transfer (the gather\n"
        "        weights are static device arrays; only host int counters live on\n"
        "        the host).\n"
        "    child BoundaryConfig: update_cadence_s = parent_dt (live-nested cadence).\n"
        "  block_until_ready between a parent step and its child recursion (schedule\n"
        "  A peak-memory bound: only one domain's transient scratch is live at a time).\n"
        "  feedback stays OFF for one-way down-nesting (Phase-3 / P0-1b+ optional).\n"
    )


__all__ = [
    "NestEdge",
    "NestTower",
    "expected_substep_counts",
    "forcedown_event_log",
    "run_host_tower",
    "runtime_hook_spec",
]
