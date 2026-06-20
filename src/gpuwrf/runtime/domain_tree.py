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

import os
import weakref

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
FusedSubstepFn = Callable[[Any, tuple[Any, ...], int, tuple[int, ...]], tuple[Any, tuple[Any, ...]]]
FusedCascadeLookup = Callable[[str], "FusedSubstepFn | None"]


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
    root_sync_cadence: int | None = None,
    edge_lookup: Callable[[DomainNest], DomainEdge] | None = None,
    fused_cascade: FusedCascadeLookup | None = None,
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
        # Output callbacks that must read carry-resident physics state (e.g. the
        # evolved Noah-MP land carry for writer-side land diagnostics) opt in via
        # a truthy ``wants_carry`` attribute and receive the FULL carry; default
        # callbacks keep the historical state-only payload.
        payload = (
            out[name]
            if getattr(output, "wants_carry", False)
            else _state_from_carry(out[name])
        )
        value = output(name, own_steps[name], payload)
        outputs.append(value)
        events.append(("output", name, own_steps[name]))

    # ---- Host-sync granularity (v0.17 nested GPU-idle fix) -------------------
    # The legacy nested path blocked on the device after EVERY single domain
    # advance (``block_between``), draining the GPU queue ~5,000x/forecast-hour
    # for the all-7 geometry; the host then walked the recursion and built the
    # next boundary packages with NOTHING queued on the GPU, so utilization
    # collapsed into ~2 s busy bursts separated by ~10 s host gaps.  The
    # parent->child ordering is a DATAFLOW dependency (the child boundary package
    # reads the just-advanced parent state; the child advance reads that forced
    # boundary), which JAX's device data-dependencies already preserve WITHOUT a
    # host block -- ``block_until_ready`` here is a memory-lifetime / queue-depth
    # policy, NOT a physics or ordering requirement.  So when ``root_sync_cadence``
    # is set we SUPPRESS the per-advance block and instead sync once per
    # ``root_sync_cadence`` completed ROOT steps.  Within a cadence window the host
    # races ahead and keeps the GPU queue full across whole root-step cascades; the
    # periodic root-step sync bounds how far the host races (peak VRAM /
    # async-dispatch depth).  events / own_steps / maybe_output / the dispatched
    # JAX ops are all UNCHANGED, so wrfout is byte-identical -- only the host WAIT
    # granularity changes.
    per_advance_block = bool(block_between) and root_sync_cadence is None
    root_cadence = int(root_sync_cadence) if root_sync_cadence is not None else 0

    def _sync_all_domains() -> None:
        leaves = [
            _state_from_carry(carry).theta
            for carry in out.values()
            if hasattr(_state_from_carry(carry), "theta")
        ]
        if leaves:
            jax.block_until_ready(leaves)

    def integrate(name: str, n_steps: int, *, reset_clock: bool, depth: int = 0) -> None:
        del reset_clock  # step clocks are per-domain global clocks, not subcycle-local
        children = hierarchy.children(name)
        if not children:
            start_step = own_steps[name] + 1
            out[name] = advance(name, out[name], int(start_step), int(n_steps))
            own_steps[name] += int(n_steps)
            events.append(("advance", name, start_step, int(n_steps), own_steps[name]))
            if per_advance_block and hasattr(_state_from_carry(out[name]), "theta"):
                jax.block_until_ready(_state_from_carry(out[name]).theta)
            maybe_output(name)
            if depth == 0 and root_cadence:
                _sync_all_domains()
            return

        cascade = fused_cascade(name) if fused_cascade is not None else None
        if cascade is not None:
            child_specs = tuple(children)
            for local in range(1, int(n_steps) + 1):
                parent_start = own_steps[name] + 1
                child_starts = tuple(own_steps[spec.child] + 1 for spec in child_specs)
                new_parent, new_children = cascade(
                    out[name],
                    tuple(out[spec.child] for spec in child_specs),
                    int(parent_start),
                    tuple(int(start) for start in child_starts),
                )

                out[name] = new_parent
                own_steps[name] += 1
                events.append(("advance", name, parent_start, 1, own_steps[name]))
                for spec in child_specs:
                    events.append(("force", spec.parent, spec.child, own_steps[spec.parent]))
                for spec, child_carry, child_start in zip(child_specs, new_children, child_starts):
                    out[spec.child] = child_carry
                    own_steps[spec.child] += int(spec.parent_grid_ratio)
                    events.append(
                        (
                            "advance",
                            spec.child,
                            child_start,
                            int(spec.parent_grid_ratio),
                            own_steps[spec.child],
                        )
                    )
                    maybe_output(spec.child)
                maybe_output(name)
                if per_advance_block:
                    _sync_all_domains()
                if depth == 0 and root_cadence and (local % root_cadence == 0 or local == int(n_steps)):
                    _sync_all_domains()
            return

        for local in range(1, int(n_steps) + 1):
            start_step = own_steps[name] + 1
            out[name] = advance(name, out[name], int(start_step), 1)
            own_steps[name] += 1
            events.append(("advance", name, start_step, 1, own_steps[name]))
            if per_advance_block and hasattr(_state_from_carry(out[name]), "theta"):
                jax.block_until_ready(_state_from_carry(out[name]).theta)
            for spec in children:
                runtime_edge = edge_lookup(spec) if edge_lookup is not None else DomainEdge(spec, weights=None)  # type: ignore[arg-type]
                if force is not None:
                    out[spec.child] = force(runtime_edge, out[spec.parent], out[spec.child])
                events.append(("force", spec.parent, spec.child, own_steps[spec.parent]))
            for spec in children:
                integrate(spec.child, int(spec.parent_grid_ratio), reset_clock=False, depth=depth + 1)
                if bool(feedback_enabled or spec.feedback) and feedback is not None:
                    runtime_edge = edge_lookup(spec) if edge_lookup is not None else DomainEdge(spec, weights=None)  # type: ignore[arg-type]
                    out[spec.parent] = feedback(runtime_edge, out[spec.parent], out[spec.child])
                    events.append(("feedback", spec.child, spec.parent, own_steps[spec.parent]))
            maybe_output(name)
            if depth == 0 and root_cadence and (local % root_cadence == 0 or local == int(n_steps)):
                _sync_all_domains()

    integrate(root_name, int(root_steps), reset_clock=False)
    # MEMORY FIX (v0.19.1): ``integrate`` is a recursive nested closure -- its
    # ``__closure__`` holds a self-reference cell (the ``integrate`` cellvar,
    # written below) AND the per-segment ``out`` carry dict.  That self-reference
    # forms a reference CYCLE reclaimable only by Python's cyclic GC, which does
    # not trip during the on-device-heavy per-output loop, so one full 9-domain
    # carry-set (~1043 device arrays) leaked per output group -> VRAM 15->30 GiB
    # -> OOM ~9.7 h on the 24 h all-7 nest.  Writing the cellvars to None severs
    # the cycle so the closures (and their captured carry copies) are freed by
    # refcount at return.  Memory-only: ``integrate`` is never called after this,
    # so numerics/speed are unchanged (proof: proofs/v019/vram_fix/).
    integrate = maybe_output = _sync_all_domains = None  # noqa: F841 - break closure ref-cycle
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


def _build_fused_cascade_program(
    *,
    parent_namelist: OperationalNamelist,
    parent_cadence: int,
    child_namelists: tuple[OperationalNamelist, ...],
    child_weights: tuple[NestForceWeights, ...],
    child_bdy_widths: tuple[int, ...],
    child_ratios: tuple[int, ...],
    child_cadences: tuple[int, ...],
) -> FusedSubstepFn:
    """Build one jitted parent-substep cascade for a flat one-way leaf subtree."""

    parent_cad = int(parent_cadence)
    n_children = len(child_namelists)

    @jax.jit
    def fused(parent_carry, child_carries, parent_start, child_starts):
        parent_new = _advance_chunk(
            parent_carry,
            parent_namelist,
            jnp.asarray(parent_start, dtype=jnp.int32),
            n_steps=1,
            cadence=parent_cad,
        )
        new_children = []
        for idx in range(n_children):
            child_carry = child_carries[idx]
            forced_state = build_child_boundary_package(
                child_carry.state,
                parent_new.state,
                child_weights[idx],
                bdy_width=int(child_bdy_widths[idx]),
            )
            child_forced = child_carry.replace(state=forced_state)
            child_new = _advance_chunk(
                child_forced,
                child_namelists[idx],
                jnp.asarray(child_starts[idx], dtype=jnp.int32),
                n_steps=int(child_ratios[idx]),
                cadence=int(child_cadences[idx]),
            )
            new_children.append(child_new)
        return parent_new, tuple(new_children)

    return fused


_FUSED_PROGRAM_CACHE: dict[int, "tuple[weakref.ref[Any], dict[str, FusedSubstepFn]]"] = {}
_FALSEY_ENV = {"0", "false", "off", "no", ""}
_TRUEY_ENV = {"1", "true", "on", "yes"}


def _env_flag(name: str) -> str | None:
    value = os.environ.get(name)
    return None if value is None else value.strip().lower()


def _nested_fuse_default_enabled() -> bool:
    """Default-on fused mode, with explicit eager opt-outs for identity proofs."""

    bitwise = _env_flag("GPUWRF_BITWISE")
    if bitwise is not None and bitwise not in _FALSEY_ENV:
        return False
    fuse = _env_flag("GPUWRF_NESTED_FUSE")
    if fuse is None:
        return True
    if fuse in _FALSEY_ENV:
        return False
    if fuse in _TRUEY_ENV:
        return True
    return True


def _fusable_parent(tree: DomainTree, name: str) -> tuple[DomainEdge, ...] | None:
    """Return edges for a safe fused parent, otherwise fail closed to eager."""

    if name == tree.hierarchy.roots()[0]:
        return None
    children = tree.hierarchy.children(name)
    if not children:
        return None
    edges: list[DomainEdge] = []
    for spec in children:
        if int(spec.parent_grid_ratio) <= 0:
            return None
        if tree.hierarchy.children(spec.child):
            return None
        if bool(spec.feedback) or bool(tree.feedback_enabled):
            return None
        edge = next((edge for edge in tree.children(name) if edge.child == spec.child), None)
        if edge is None or edge.weights is None:
            return None
        edges.append(edge)
    return tuple(edges)


def _operational_fused_cascade_factory(tree: DomainTree) -> FusedCascadeLookup:
    """Build a default-on fused cascade lookup, fail-closed when unsafe."""

    enabled = _nested_fuse_default_enabled()
    cache: dict[str, FusedSubstepFn | None] = {}

    def lookup(parent_name: str) -> FusedSubstepFn | None:
        if not enabled:
            return None
        if parent_name in cache:
            return cache[parent_name]
        edges = _fusable_parent(tree, parent_name)
        if edges is None:
            cache[parent_name] = None
            return None

        parent_namelist = tree.domains[parent_name].namelist
        parent_cadence = int(parent_namelist.radiation_cadence_steps)
        child_namelists = tuple(tree.domains[edge.child].namelist for edge in edges)
        child_cadences = tuple(int(namelist.radiation_cadence_steps) for namelist in child_namelists)
        if parent_cadence <= 0 or any(cadence <= 0 for cadence in child_cadences):
            cache[parent_name] = None
            return None

        key = id(tree)
        entry = _FUSED_PROGRAM_CACHE.get(key)
        if entry is not None and entry[0]() is tree:
            tree_programs = entry[1]
        else:
            tree_programs = {}
            _FUSED_PROGRAM_CACHE[key] = (
                weakref.ref(tree, lambda _ref, _key=key: _FUSED_PROGRAM_CACHE.pop(_key, None)),
                tree_programs,
            )

        program = tree_programs.get(str(parent_name))
        if program is None:
            program = _build_fused_cascade_program(
                parent_namelist=parent_namelist,
                parent_cadence=parent_cadence,
                child_namelists=child_namelists,
                child_weights=tuple(edge.weights for edge in edges),
                child_bdy_widths=tuple(int(tree.domains[edge.child].state.u_bdy.shape[2]) for edge in edges),
                child_ratios=tuple(int(edge.parent_grid_ratio) for edge in edges),
                child_cadences=child_cadences,
            )
            tree_programs[str(parent_name)] = program
        cache[parent_name] = program
        return program

    return lookup


def _operational_feedback(edge: DomainEdge, parent: OperationalCarry, child: OperationalCarry) -> OperationalCarry:
    if edge.feedback_weights is None:
        return parent
    # NOTE (v0.13 VRAM).  The feedback runs EAGERLY (op-by-op), NOT jitted, on
    # purpose: a measured A/B on the 9/3/1 km d02->d03 edge showed the eager path
    # peaks LOWER than a fused ``jax.jit`` feedback program.  Eager dispatch keeps
    # only ONE leaf's gather/scatter/smoother scratch live at a time (each
    # ``apply_state_feedback`` per-leaf result is consumed and freed before the next
    # leaf), whereas XLA's buffer-assignment schedules several leaves' transients
    # concurrently, raising the simultaneous working set.  Donating the parent
    # state is also unsafe here: the carried WRF ``*_save`` scratch aliases the
    # parent ``state`` leaves (``u_save = state.u`` etc.), which must stay valid for
    # the next parent advance.  The peak VRAM reduction therefore comes from inside
    # ``apply_state_feedback`` (rebuilding each p/ph/mu total ONCE and sharing it
    # with its legacy alias, removing 6 redundant full-parent-field temporaries),
    # measured bit-identical; see proofs/v013/twoway_vram.*.
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
    root_sync_cadence: int | None = None,
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

    effective_feedback = tree.feedback_enabled if feedback_enabled is None else bool(feedback_enabled)
    fused_cascade = None if effective_feedback else _operational_fused_cascade_factory(tree)

    return run_domain_tree_callbacks(
        tree.hierarchy,
        carries,
        root_steps=int(root_steps),
        advance=_operational_advance_factory(tree),
        force=_operational_force,
        feedback=_operational_feedback,
        root=root,
        feedback_enabled=effective_feedback,
        output=output,
        output_cadence_steps=output_cadence_steps,
        block_between=block_between,
        root_sync_cadence=root_sync_cadence,
        edge_lookup=lookup,
        fused_cascade=fused_cascade,
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
