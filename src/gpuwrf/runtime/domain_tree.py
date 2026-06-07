"""Live multi-domain nested runtime orchestration.

The runtime mirrors WRF's ``module_integrate`` nesting cadence while leaving the
single-domain timestep implementation in ``operational_mode`` untouched:

1. advance the parent one of its own timesteps;
2. build each child boundary package from the just-advanced live parent state;
3. recurse the child for ``parent_grid_ratio`` substeps;
4. optionally feed the child overlap back to the parent behind an explicit gate.

The host loop carries opaque device-resident ``OperationalCarry`` objects.  The
default operational adapter calls the existing ``_advance_chunk`` segmented
single-domain entry, so the WRF scratch carry persists across nested substeps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest, DycoreMetrics, GridSpec
from gpuwrf.contracts.state import State
from gpuwrf.coupling.boundary_apply import BoundaryConfig
from gpuwrf.coupling.boundary_feedback import (
    StateFeedbackWeights,
    apply_state_feedback,
    build_state_feedback_weights,
)
from gpuwrf.nesting.boundary_construction import (
    NestForceWeights,
    build_child_boundary_package,
    build_nest_force_weights,
)
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _advance_chunk,
    _initial_carry_for_run,
    _resolve_operational_suite,
)
from gpuwrf.runtime.operational_state import OperationalCarry


AdvanceFn = Callable[[str, Any, int, int], Any]
ForceFn = Callable[["DomainEdge", Any, Any], Any]
FeedbackFn = Callable[["DomainEdge", Any, Any], Any]
OutputFn = Callable[[str, int, Any], Any]


@dataclass(frozen=True)
class DomainBundle:
    """Per-domain runtime bundle: state, namelist, grid, and metrics."""

    name: str
    state: State
    namelist: OperationalNamelist
    grid: GridSpec | None = None
    metrics: DycoreMetrics | None = None

    def __post_init__(self) -> None:
        grid = self.grid if self.grid is not None else self.namelist.grid
        metrics = self.metrics if self.metrics is not None else self.namelist.metrics
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "metrics", metrics)


@dataclass(frozen=True)
class DomainEdge:
    """Runtime edge: static nest metadata plus device gather plans."""

    spec: DomainNest
    weights: NestForceWeights
    feedback_weights: StateFeedbackWeights | None = None

    @property
    def parent(self) -> str:
        return self.spec.parent

    @property
    def child(self) -> str:
        return self.spec.child

    @property
    def parent_grid_ratio(self) -> int:
        return int(self.spec.parent_grid_ratio)


@dataclass(frozen=True)
class DomainTree:
    """A live nested tower with per-domain bundles and edge operators."""

    hierarchy: DomainHierarchy
    domains: dict[str, DomainBundle]
    edges: dict[str, tuple[DomainEdge, ...]] = field(default_factory=dict)
    feedback_enabled: bool = False

    @classmethod
    def from_domains(
        cls,
        hierarchy: DomainHierarchy,
        domains: dict[str, DomainBundle],
        *,
        registration: str = "sint",
        feedback_enabled: bool = False,
        feedback_spec_zone: int = 1,
    ) -> "DomainTree":
        """Build a runtime tree and precompute all forcedown/feedback weights."""

        missing = [name for name in hierarchy.order if name not in domains]
        if missing:
            raise ValueError(f"missing DomainBundle(s) for {missing}")
        edges_by_parent: dict[str, list[DomainEdge]] = {}
        for edge in hierarchy.nests:
            parent = domains[edge.parent]
            child = domains[edge.child]
            if parent.grid is None or child.grid is None:
                raise ValueError(f"edge {edge.parent}->{edge.child} requires parent and child grids")
            weights = build_nest_force_weights(
                parent_grid_ratio=edge.parent_grid_ratio,
                i_parent_start=edge.i_parent_start,
                j_parent_start=edge.j_parent_start,
                parent_grid=parent.grid,
                child_grid=child.grid,
                registration=registration,
            )
            # Precompute feedback weights unconditionally: the device op remains
            # behind the runtime gate, but a manager can enable two-way feedback
            # after construction without silently getting a no-op edge.
            feedback_weights = build_state_feedback_weights(
                parent_grid_ratio=edge.parent_grid_ratio,
                i_parent_start=edge.i_parent_start,
                j_parent_start=edge.j_parent_start,
                parent_grid=parent.grid,
                child_grid=child.grid,
                spec_zone=int(feedback_spec_zone),
            )
            edges_by_parent.setdefault(edge.parent, []).append(
                DomainEdge(spec=edge, weights=weights, feedback_weights=feedback_weights)
            )
        return cls(
            hierarchy=hierarchy,
            domains=dict(domains),
            edges={name: tuple(items) for name, items in edges_by_parent.items()},
            feedback_enabled=bool(feedback_enabled),
        )

    def children(self, parent: str) -> tuple[DomainEdge, ...]:
        return self.edges.get(parent, ())

    def root(self) -> str:
        return self.hierarchy.roots()[0]

    def persistent_state_bytes(self) -> dict[str, int]:
        return {name: int(bundle.state.bytes()) for name, bundle in self.domains.items()}


@dataclass(frozen=True)
class DomainTreeResult:
    """Final carries/states plus a concise event/output audit."""

    carries: dict[str, Any]
    states: dict[str, Any]
    own_steps: dict[str, int]
    events: tuple[tuple[Any, ...], ...]
    outputs: tuple[Any, ...]


def build_live_nested_boundary_config(
    parent_dt_s: float,
    *,
    spec_bdy_width: int = 5,
    spec_zone: int = 1,
    relax_zone: int = 4,
    nested_ph_relax: bool = True,
    nested_w_relax: bool = True,
    nested_ph_spec: bool = True,
) -> BoundaryConfig:
    """Boundary config for a live child edge: cadence equals the parent timestep."""

    return BoundaryConfig(
        spec_bdy_width=int(spec_bdy_width),
        spec_zone=int(spec_zone),
        relax_zone=int(relax_zone),
        update_cadence_s=float(parent_dt_s),
        force_geopotential=False,
        nested_ph_relax=bool(nested_ph_relax),
        nested_w_relax=bool(nested_w_relax),
        nested_ph_spec=bool(nested_ph_spec),
    )


def with_live_child_boundary_config(
    namelist: OperationalNamelist,
    *,
    parent_dt_s: float,
    nested_ph_relax: bool = True,
    nested_w_relax: bool = True,
    nested_ph_spec: bool = True,
) -> OperationalNamelist:
    """Return ``namelist`` with WRF live-nest child boundary cadence/toggles."""

    cfg = build_live_nested_boundary_config(
        parent_dt_s,
        spec_bdy_width=int(namelist.boundary_config.spec_bdy_width),
        spec_zone=int(namelist.boundary_config.spec_zone),
        relax_zone=int(namelist.boundary_config.relax_zone),
        nested_ph_relax=nested_ph_relax,
        nested_w_relax=nested_w_relax,
        nested_ph_spec=nested_ph_spec,
    )
    return dataclass_replace(namelist, boundary_config=cfg)


def _state_from_carry(carry: Any) -> Any:
    return getattr(carry, "state", carry)


def run_domain_tree_callbacks(
    hierarchy: DomainHierarchy,
    carries: dict[str, Any],
    *,
    root_steps: int,
    advance: AdvanceFn,
    force: ForceFn | None = None,
    feedback: FeedbackFn | None = None,
    root: str | None = None,
    feedback_enabled: bool = False,
    output: OutputFn | None = None,
    output_cadence_steps: dict[str, int] | None = None,
    block_between: bool = True,
    edge_lookup: Callable[[DomainNest], DomainEdge] | None = None,
    initial_own_steps: dict[str, int] | None = None,
) -> DomainTreeResult:
    """Generic WRF-recursive domain-tree runner.

    This function is deliberately callback-based so CPU unit gates can exercise
    the cadence without constructing real ``State`` objects.  The operational
    wrapper below binds these callbacks to ``_advance_chunk`` and
    ``build_child_boundary_package``.

    ``initial_own_steps`` seeds each domain's GLOBAL step clock so the runner can
    be driven in contiguous output-interval segments from a host loop (each
    segment carries the prior segment's carries + own_steps).  The in-chunk
    radiation gate keys off the global ``start_step`` index, so threading the
    clock across segments makes the segmented run fire radiation on exactly the
    same global steps as a single full-length call -- the memory-bounded nested
    analogue of ``run_forecast_operational_segmented`` (peak VRAM independent of
    forecast length).  Defaults to ``0`` for every domain (a fresh full run).
    """

    root_name = root or hierarchy.roots()[0]
    out = dict(carries)
    events: list[tuple[Any, ...]] = []
    outputs: list[Any] = []
    own_steps = {name: 0 for name in hierarchy.order}
    if initial_own_steps:
        for name, value in initial_own_steps.items():
            if name in own_steps:
                own_steps[name] = int(value)
    output_cadence_steps = dict(output_cadence_steps or {})

    def maybe_output(name: str) -> None:
        cadence = int(output_cadence_steps.get(name, 0))
        if output is None or cadence <= 0 or own_steps[name] % cadence != 0:
            return
        value = output(name, own_steps[name], _state_from_carry(out[name]))
        outputs.append(value)
        events.append(("output", name, own_steps[name]))

    def integrate(name: str, n_steps: int, *, reset_clock: bool) -> None:
        del reset_clock  # step clocks are per-domain global clocks, not subcycle-local
        children = hierarchy.children(name)
        if not children:
            start_step = own_steps[name] + 1
            out[name] = advance(name, out[name], int(start_step), int(n_steps))
            own_steps[name] += int(n_steps)
            events.append(("advance", name, start_step, int(n_steps), own_steps[name]))
            if block_between and hasattr(_state_from_carry(out[name]), "theta"):
                jax.block_until_ready(_state_from_carry(out[name]).theta)
            maybe_output(name)
            return

        for local in range(1, int(n_steps) + 1):
            start_step = own_steps[name] + 1
            out[name] = advance(name, out[name], int(start_step), 1)
            own_steps[name] += 1
            events.append(("advance", name, start_step, 1, own_steps[name]))
            if block_between and hasattr(_state_from_carry(out[name]), "theta"):
                jax.block_until_ready(_state_from_carry(out[name]).theta)
            for spec in children:
                runtime_edge = edge_lookup(spec) if edge_lookup is not None else DomainEdge(spec, weights=None)  # type: ignore[arg-type]
                if force is not None:
                    out[spec.child] = force(runtime_edge, out[spec.parent], out[spec.child])
                events.append(("force", spec.parent, spec.child, own_steps[spec.parent]))
            for spec in children:
                integrate(spec.child, int(spec.parent_grid_ratio), reset_clock=False)
                if bool(feedback_enabled or spec.feedback) and feedback is not None:
                    runtime_edge = edge_lookup(spec) if edge_lookup is not None else DomainEdge(spec, weights=None)  # type: ignore[arg-type]
                    out[spec.parent] = feedback(runtime_edge, out[spec.parent], out[spec.child])
                    events.append(("feedback", spec.child, spec.parent, own_steps[spec.parent]))
            maybe_output(name)

    integrate(root_name, int(root_steps), reset_clock=False)
    states = {name: _state_from_carry(carry) for name, carry in out.items()}
    return DomainTreeResult(
        carries=out,
        states=states,
        own_steps=own_steps,
        events=tuple(events),
        outputs=tuple(outputs),
    )


def _operational_advance_factory(tree: DomainTree) -> AdvanceFn:
    def advance(name: str, carry: OperationalCarry, start_step: int, n_steps: int) -> OperationalCarry:
        namelist = tree.domains[name].namelist
        cadence = int(namelist.radiation_cadence_steps)
        if cadence <= 0:
            raise ValueError(f"{name}: radiation_cadence_steps must be positive")
        return _advance_chunk(
            carry,
            namelist,
            jnp.asarray(int(start_step), dtype=jnp.int32),
            n_steps=int(n_steps),
            cadence=cadence,
        )

    return advance


def _operational_force(edge: DomainEdge, parent: OperationalCarry, child: OperationalCarry) -> OperationalCarry:
    child_state = child.state
    bdy_width = int(child_state.u_bdy.shape[2])
    forced_state = build_child_boundary_package(
        child_state,
        parent.state,
        edge.weights,
        bdy_width=bdy_width,
    )
    return child.replace(state=forced_state)


def _operational_feedback(edge: DomainEdge, parent: OperationalCarry, child: OperationalCarry) -> OperationalCarry:
    if edge.feedback_weights is None:
        return parent
    fed = apply_state_feedback(parent.state, child.state, edge.feedback_weights, feedback=True)
    return parent.replace(state=fed)


def run_operational_domain_tree(
    tree: DomainTree,
    *,
    root_steps: int,
    root: str | None = None,
    feedback_enabled: bool | None = None,
    output: OutputFn | None = None,
    output_cadence_steps: dict[str, int] | None = None,
    block_between: bool = True,
    carries: dict[str, Any] | None = None,
    initial_own_steps: dict[str, int] | None = None,
) -> DomainTreeResult:
    """Run a live nested operational tree for ``root_steps`` root timesteps.

    Pass ``carries`` (a prior :class:`DomainTreeResult`'s ``carries``) and
    ``initial_own_steps`` (its ``own_steps``) to RESUME from a previous segment.
    This lets a host loop drive the live nest in contiguous output-interval
    segments, ``block_until_ready``-ing + freeing each segment's device scratch
    before the next allocates -- bounding peak VRAM independent of forecast
    length while keeping the recursion cadence + radiation schedule byte-identical
    to a single full-length call.  When ``carries`` is ``None`` the initial
    carries are built fresh from each domain's bundle state (a cold start).
    """

    for name, bundle in tree.domains.items():
        _resolve_operational_suite(bundle.namelist)
        if int(bundle.namelist.rk_order) != 3:
            raise ValueError(f"{name}: operational nesting currently supports RK3 only")
    if carries is None:
        carries = {
            name: _initial_carry_for_run(bundle.state, bundle.namelist)
            for name, bundle in tree.domains.items()
        }
    else:
        carries = dict(carries)
    edge_by_pair = {
        (edge.parent, edge.child): edge
        for edges in tree.edges.values()
        for edge in edges
    }

    def lookup(spec: DomainNest) -> DomainEdge:
        return edge_by_pair[(spec.parent, spec.child)]

    return run_domain_tree_callbacks(
        tree.hierarchy,
        carries,
        root_steps=int(root_steps),
        advance=_operational_advance_factory(tree),
        force=_operational_force,
        feedback=_operational_feedback,
        root=root,
        feedback_enabled=tree.feedback_enabled if feedback_enabled is None else bool(feedback_enabled),
        output=output,
        output_cadence_steps=output_cadence_steps,
        block_between=block_between,
        edge_lookup=lookup,
        initial_own_steps=initial_own_steps,
    )


__all__ = [
    "DomainBundle",
    "DomainEdge",
    "DomainTree",
    "DomainTreeResult",
    "build_live_nested_boundary_config",
    "run_domain_tree_callbacks",
    "run_operational_domain_tree",
    "with_live_child_boundary_config",
]
