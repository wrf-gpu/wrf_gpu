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
import sys
import weakref

from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
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
    _advance_chunk_fori,
    _initial_carry_for_run,
    _resolve_operational_suite,
    build_clock_base,
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
    """Final carries/states plus a concise event/output audit.

    ``events`` / ``outputs`` are the full per-call audit lists (unbounded within a
    single call, but a single call only spans ``root_steps`` -- the segmented host
    loop drives one output interval per call).  When the caller opts in via
    ``max_event_tail`` (see :func:`run_domain_tree_callbacks`) these become the
    BOUNDED most-recent tail and the aggregate moves to the summary fields below:

    * ``event_counts`` -- total count by event TYPE (``advance``/``force``/
      ``feedback``/``output``).  Empty when no cap is set.
    * ``force_counts`` -- count by ``parent->child`` force edge.  Empty when no cap.

    The summary fields default empty so every existing consumer (and the
    bit-identical fp64 path, which sets no cap) is unaffected.
    """

    carries: dict[str, Any]
    states: dict[str, Any]
    own_steps: dict[str, int]
    events: tuple[tuple[Any, ...], ...]
    outputs: tuple[Any, ...]
    event_counts: dict[str, int] = field(default_factory=dict)
    force_counts: dict[str, int] = field(default_factory=dict)


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


class _BoundedTail:
    """A list-like ``.append``/iterable that keeps only the last ``cap`` items.

    Drop-in for the plain ``outputs`` list when the opt-in ``max_event_tail`` host
    guard is on.  ``tuple(...)`` and iteration yield the retained tail in order.
    """

    __slots__ = ("_tail",)

    def __init__(self, cap: int) -> None:
        self._tail: deque = deque(maxlen=int(cap))

    def append(self, item: Any) -> None:
        self._tail.append(item)

    def __iter__(self):
        return iter(self._tail)

    def __len__(self) -> int:
        return len(self._tail)


class _BoundedEventLog(_BoundedTail):
    """Bounded event tail that ALSO folds an exact aggregate as it appends.

    The retained tail is the last ``cap`` event tuples (for a post-mortem); the
    ``counts`` / ``force_counts`` Counters are the EXACT lifetime aggregate by
    event type and force edge -- identical to counting the full (un-capped) log,
    because counting is order-independent.  This lets a single long call report a
    complete audit summary with O(cap) host RAM instead of O(events).
    """

    __slots__ = ("counts", "force_counts")

    def __init__(self, cap: int) -> None:
        super().__init__(cap)
        self.counts: Counter = Counter()
        self.force_counts: Counter = Counter()

    def append(self, item: tuple[Any, ...]) -> None:
        super().append(item)
        if not item:
            return
        self.counts[item[0]] += 1
        if item[0] == "force":
            self.force_counts[f"{item[1]}->{item[2]}"] += 1


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
    max_event_tail: int | None = None,
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

    ``max_event_tail`` is an OPT-IN host-RAM guard (default ``None`` =
    unbounded = bit-identical legacy behaviour).  When set to a positive int the
    ``events`` / ``outputs`` audit lists are capped to that many most-recent
    entries (a ``deque`` tail) while the aggregate is folded into the result's
    ``event_counts`` / ``force_counts`` summary dicts.  ``0`` is equivalent to
    ``None`` (unbounded).  The per-CALL lists are already bounded by ``root_steps``
    so the default path never needs this; it exists for callers that drive ONE
    long call (rather than the segmented host loop) and want O(1) host RAM in
    forecast length.  The cap changes NO dispatched op, callback, or step clock --
    only how many host audit tuples are retained -- so numerics stay identical.
    """

    root_name = root or hierarchy.roots()[0]
    out = dict(carries)
    _cap = int(max_event_tail) if max_event_tail is not None else 0
    # ``events`` / ``outputs`` are append-only audit logs.  Default = plain lists
    # (unbounded, bit-identical).  When the opt-in cap is on they become bounded
    # tails that fold an exact aggregate (Counter) as they grow -- O(cap) host RAM.
    events: Any
    outputs: Any
    if _cap > 0:
        events = _BoundedEventLog(_cap)
        outputs = _BoundedTail(_cap)
    else:
        events = []
        outputs = []
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
    if _cap > 0:
        return DomainTreeResult(
            carries=out,
            states=states,
            own_steps=own_steps,
            events=tuple(events),
            outputs=tuple(outputs),
            event_counts=dict(events.counts),
            force_counts=dict(events.force_counts),
        )
    return DomainTreeResult(
        carries=out,
        states=states,
        own_steps=own_steps,
        events=tuple(events),
        outputs=tuple(outputs),
    )


def _nested_aot_enabled() -> bool:
    """``GPUWRF_NESTED_AOT`` truthy? DEFAULT-ON (vNext cheap-key manifest).

    Step C: when ON, the eager advance closure computes a metadata-only
    ``cheap_key`` for the exact per-domain/per-shape call (NO lowering) and loads
    the matching serialized ``_advance_chunk_fori`` blob in microseconds; on a
    miss it compiles + serializes that exact variant (under both the cheap_key and
    the hlo address). Identity-preserving + fail-open: any error falls back to the
    normal jitted compile -- never wrong, only slower. Set ``GPUWRF_NESTED_AOT=0``
    to disable. The version-fingerprint guard (target/jaxlib/SM/x64/XLA_FLAGS) is
    re-checked on every load so a stale-target blob is never mis-executed."""
    return os.environ.get("GPUWRF_NESTED_AOT", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _nested_aot_verify_enabled() -> bool:
    """``GPUWRF_AOT_VERIFY`` truthy? (default OFF).

    Verify-mode: on the FIRST cheap-key hit per (domain, cheap_key) per process,
    do ONE confirmatory ``lower()`` and assert the lowered HLO digest equals the
    blob's recorded ``meta.hlo_sha256``. PASS -> mark verified, never lower that
    key again this process. FAIL -> fail-CLOSED for that key (discard the blob,
    compile normally, loud stderr) so any missed cheap_key determinant becomes a
    loud fallback instead of a silent wrong result. Cost = one lower per distinct
    key per process (not per chunk). The manager runs this ON for the first
    warm-start of each new version, then OFF in production."""
    return os.environ.get("GPUWRF_AOT_VERIFY", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Records the last AOT-load outcome per domain (diagnostics / GPU A/B). Maps
# domain name -> {"loaded": bool, "source": str, "error": str|None, ...}.
# Numerically inert.
NESTED_AOT_STATUS: dict[str, object] = {"enabled": False, "domains": {}}


def _operational_advance_factory(tree: DomainTree) -> AdvanceFn:
    # #91: one traced clock_base per domain namelist (built once, reused every chunk)
    # so each domain's compiled HLO is date-independent (cross-date cache hit).
    _clock_bases = {
        name: build_clock_base(dom.namelist) for name, dom in tree.domains.items()
    }

    # Step C: per-domain/per-HLO AOT-loaded advance callables. Integration can
    # call one domain with multiple carry-shape variants; memoizing only by domain
    # would load one blob and then false-fallback every sibling shape.
    aot_on = _nested_aot_enabled()
    aot_verify = _nested_aot_verify_enabled()
    NESTED_AOT_STATUS["enabled"] = aot_on
    NESTED_AOT_STATUS["verify"] = aot_verify
    NESTED_AOT_STATUS["domains"] = {}
    # Memo of loaded advance callables keyed by (domain, cheap_key). Integration
    # can call one domain with multiple carry-shape variants, so memoizing only by
    # domain would load one blob and then false-fallback every sibling shape.
    _aot_calls: dict[tuple[str, str], Any] = {}
    _aot_attempted: set[tuple[str, str]] = set()
    # Verify-mode bookkeeping: keys whose blob HLO has been confirmed (lower-once)
    # and keys proven to FAIL verification (fail-closed -> never load again).
    _aot_verified: set[tuple[str, str]] = set()
    _aot_verify_failed: set[tuple[str, str]] = set()

    def _record_aot_status(name: str, status: dict[str, Any]) -> None:
        try:
            NESTED_AOT_STATUS["domains"][name] = dict(status)  # type: ignore[index]
        except Exception:  # noqa: BLE001
            pass

    def _log_aot_status(name: str, status: dict[str, Any]) -> None:
        """Emit one stderr line per domain load/fallback for GPU gate visibility."""
        try:
            loaded = bool(status.get("loaded"))
            source = status.get("source") or ("aot_blob" if loaded else "fallback:jit")
            parts = [
                f"[gpuwrf:nested-aot] domain={name}",
                f"loaded={str(loaded).lower()}",
                f"source={source}",
            ]
            blob_path = status.get("blob_path")
            if blob_path:
                parts.append(f"path={blob_path}")
            hlo = status.get("hlo_sha256")
            if hlo:
                parts.append(f"hlo={str(hlo)[:12]}")
            error = status.get("error")
            if error:
                parts.append(f"error={str(error).replace(chr(10), ' | ')}")
            sys.stderr.write(" ".join(parts) + "\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass

    def _aot_fori_mode_enabled() -> bool:
        mode = os.environ.get("GPUWRF_ADVANCE_CHUNK_LOOP", "fori").strip().lower()
        return mode in {"", "fori", "fori_loop", "fori-loop"}

    def _cheap_key_for(
        carry: OperationalCarry,
        namelist: OperationalNamelist,
        start: Any,
        clock_base: Any,
        *,
        n_steps: int,
        cadence: int,
    ) -> str | None:
        """Compute the metadata-only cheap_key for this call (NO lowering).

        Returns the key, or ``None`` on any error (caller falls back to the
        lower-only path -> compile). This is the microsecond replacement for the
        ~30-54 min ``_lower_advance_variant`` on the WARM path."""
        try:
            from gpuwrf.runtime import aot_cheap_key as _ck

            return _ck.cheap_key(
                _advance_chunk_fori,
                (carry, namelist, start, clock_base),
                {"n_steps": int(n_steps), "cadence": int(cadence)},
                namelist,
            )
        except BaseException:  # noqa: BLE001 - fail-open to the lower path
            return None

    def _lower_advance_variant(
        carry: OperationalCarry,
        namelist: OperationalNamelist,
        start: Any,
        clock_base: Any,
        *,
        n_steps: int,
        cadence: int,
    ) -> tuple[Any | None, str | None, str | None]:
        """Lower the exact Step-C call and return ``(lowered, hlo, error)``."""
        try:
            from gpuwrf.runtime import aot_executable as aotx

            lowered = _advance_chunk_fori.lower(
                carry,
                namelist,
                start,
                clock_base,
                n_steps=int(n_steps),
                cadence=int(cadence),
            )
            hlo = aotx.hlo_sha256_from_lowered(lowered)
            if not hlo:
                return lowered, None, "lowered HLO digest unavailable"
            return lowered, hlo, None
        except BaseException as exc:  # noqa: BLE001 - fail-open to jitted path
            return None, None, f"{type(exc).__name__}: {exc}"

    def _verify_cheap_key_blob(
        name: str,
        ckey: str,
        meta_hlo_sha256: str | None,
        carry: OperationalCarry,
        namelist: OperationalNamelist,
        start: Any,
        clock_base: Any,
        *,
        n_steps: int,
        cadence: int,
    ) -> bool:
        """Verify a cheap-key-loaded blob lowers to its recorded HLO (lower-once).

        Returns ``True`` (mark verified, never lower this key again) when the live
        lowered HLO digest equals ``meta_hlo_sha256``; ``False`` (fail-CLOSED, loud
        stderr) on ANY mismatch / missing digest / lower error so a missed cheap_key
        determinant surfaces as a fallback rather than a silent wrong result. Cost =
        one lower per distinct (domain, cheap_key) per process, not per chunk."""
        ck_state = (name, ckey)
        _, hlo_now, lower_error = _lower_advance_variant(
            carry,
            namelist,
            start,
            clock_base,
            n_steps=int(n_steps),
            cadence=int(cadence),
        )
        if lower_error or not hlo_now or not meta_hlo_sha256:
            _aot_verify_failed.add(ck_state)
            reason = lower_error or "missing HLO digest for verify"
            status = {
                "name": name,
                "loaded": False,
                "source": "fallback:verify-error",
                "error": f"AOT verify could not confirm cheap_key blob: {reason}",
                "cheap_key": ckey,
            }
            _record_aot_status(name, status)
            _log_aot_status(name, status)
            return False
        if hlo_now != meta_hlo_sha256:
            # CHEAP_KEY COLLISION: the blob this key located lowers to a DIFFERENT
            # HLO -> the key missed a determinant. Fail CLOSED + loud, and POISON
            # the (domain, cheap_key) on disk (P0-1) so NO process ever writes or
            # serves a cheap-key blob for this ambiguous key again -- the next load
            # fails OPEN to a fresh compile instead of overwriting/serving a sibling.
            _aot_verify_failed.add(ck_state)
            try:
                from gpuwrf.runtime import aot_precompile as _aotp

                _aotp.quarantine_cheap_key(
                    name,
                    ckey,
                    reason="verify-mode HLO mismatch (cheap_key collision)",
                    detail={
                        "meta_hlo": str(meta_hlo_sha256),
                        "live_hlo": str(hlo_now),
                    },
                )
            except Exception:  # noqa: BLE001 - quarantine is best-effort/fail-open
                pass
            status = {
                "name": name,
                "loaded": False,
                "source": "fallback:verify-MISMATCH",
                "error": (
                    "AOT VERIFY FAILED (cheap_key collision): blob meta hlo="
                    f"{str(meta_hlo_sha256)[:12]} != live lowered hlo="
                    f"{str(hlo_now)[:12]}; quarantining cheap_key + compiling fresh "
                    "(cheap_key is missing an HLO determinant)"
                ),
                "cheap_key": ckey,
            }
            _record_aot_status(name, status)
            _log_aot_status(name, status)
            return False
        _aot_verified.add(ck_state)
        return True

    def _aot_advance_by_cheap_key(name: str, ckey: str):
        """Return ``(call, status)`` for ``name``/``cheap_key`` via metadata lookup.

        Loads the cheap-key-addressed blob WITHOUT lowering (memoised per
        (domain, cheap_key)). Fully fail-open: any error yields ``(None, status)``
        and the caller compiles/captures. ``status`` carries ``meta_hlo_sha256``
        so verify-mode can confirm the loaded blob lowers to the same HLO."""
        key = (name, ckey)
        status: dict[str, Any] = {
            "name": name,
            "loaded": False,
            "source": "fallback:jit",
            "error": None,
            "cheap_key": ckey,
        }
        if key in _aot_attempted:
            return _aot_calls.get(key), status
        _aot_attempted.add(key)
        call = None
        try:
            from gpuwrf.runtime import aot_precompile

            loaded = aot_precompile.load_domain_blob(
                name, cheap_key=ckey, return_status=True
            )
            if isinstance(loaded, tuple) and len(loaded) == 2:
                call, loaded_status = loaded
                if isinstance(loaded_status, dict):
                    status.update(loaded_status)
            else:  # back-compat monkeypatch returning a bare call/None
                call = loaded
                status.update(
                    {
                        "loaded": call is not None,
                        "source": "aot_blob" if call is not None else "fallback:jit",
                    }
                )
        except BaseException as exc:  # noqa: BLE001 - fail-open
            call = None
            status.update(
                {
                    "loaded": False,
                    "source": "fallback:load-exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        _aot_calls[key] = call
        status["loaded"] = call is not None
        # Surface the blob's recorded HLO as ``hlo_sha256`` too so the diagnostic
        # log line / report still shows WHICH program loaded (cheap_key is the
        # lookup address; the HLO is the program identity).
        if status.get("hlo_sha256") is None and status.get("meta_hlo_sha256"):
            status["hlo_sha256"] = status["meta_hlo_sha256"]
        return call, status

    def _aot_advance_for(name: str, hlo_sha256: str):
        """Return a drop-in AOT advance callable for ``name``/``hlo`` or ``None``.

        Loaded at most once per domain variant (memoised). Fully fail-open: any
        error in the load path yields ``None`` and the caller compiles/captures."""
        key = (name, hlo_sha256)
        if key in _aot_attempted:
            return _aot_calls.get(key)
        _aot_attempted.add(key)
        call = None
        status: dict[str, Any] = {
            "name": name,
            "loaded": False,
            "source": "fallback:jit",
            "error": None,
            "hlo_sha256": hlo_sha256,
        }
        try:
            from gpuwrf.runtime import aot_precompile

            loaded = aot_precompile.load_domain_blob(
                name, hlo_sha256=hlo_sha256, return_status=True
            )
            if isinstance(loaded, tuple) and len(loaded) == 2:
                call, loaded_status = loaded
                if isinstance(loaded_status, dict):
                    status.update(loaded_status)
            else:
                # Back-compat for tests/older monkeypatches that return just call/None.
                call = loaded
                status.update(
                    {
                        "loaded": call is not None,
                        "source": "aot_blob" if call is not None else "fallback:jit",
                    }
                )
        except BaseException as exc:  # noqa: BLE001 - fail-open
            call = None
            status.update(
                {
                    "loaded": False,
                    "source": "fallback:load-exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        _aot_calls[key] = call
        status["loaded"] = call is not None
        if call is not None and not status.get("source"):
            status["source"] = "aot_blob"
        _record_aot_status(name, status)
        _log_aot_status(name, status)
        return call

    def _compile_capture_and_call(
        name: str,
        lowered: Any | None,
        hlo_sha256: str | None,
        carry: OperationalCarry,
        namelist: OperationalNamelist,
        start: Any,
        clock_base: Any,
        *,
        n_steps: int,
        cadence: int,
        cheap_key: str | None = None,
    ) -> OperationalCarry:
        """Compile the exact variant, serialize it best-effort, then execute it.

        On a cheap-key miss (or verify fail-closed) this compiles the lowered
        program and serializes the blob under BOTH the cheap_key address (so the
        next warm process locates it without lowering) and the hlo address."""
        if lowered is None:
            return _advance_chunk(
                carry,
                namelist,
                start,
                clock_base,
                n_steps=int(n_steps),
                cadence=cadence,
            )
        try:
            compiled = lowered.compile()
            if hlo_sha256:
                _aot_attempted.add((name, hlo_sha256))
                _aot_calls[(name, hlo_sha256)] = compiled
            if cheap_key:
                _aot_attempted.add((name, cheap_key))
                _aot_calls[(name, cheap_key)] = compiled
            capture_status: dict[str, Any] = {
                "name": name,
                "loaded": False,
                "source": "fallback:jit-compiled",
                "error": None,
                "hlo_sha256": hlo_sha256,
                "cheap_key": cheap_key,
            }
            if hlo_sha256 or cheap_key:
                try:
                    from gpuwrf.runtime import aot_precompile

                    key_schema = None
                    if cheap_key:
                        try:
                            from gpuwrf.runtime import aot_cheap_key as _ck

                            key_schema = _ck.KEY_SCHEMA
                        except Exception:  # noqa: BLE001
                            key_schema = None
                    ser = aot_precompile._serialize_domain_blob(
                        name,
                        compiled,
                        None,
                        hlo_sha256=hlo_sha256,
                        # Thread the lowered object so serialize() can re-derive the
                        # lower-only StableHLO digest even if hlo_sha256 came back
                        # None on this backend -> persisted meta.hlo_sha256 is never
                        # empty -> the cross-process cheap-key load + verify succeed.
                        lowered=lowered,
                        cheap_key=cheap_key,
                        key_schema=key_schema,
                    )
                    capture_status.update(ser)
                    if ser.get("aot_written"):
                        capture_status["source"] = "fallback:jit-compiled+aot-captured"
                    elif ser.get("aot_error"):
                        capture_status["error"] = ser.get("aot_error")
                except BaseException as exc:  # noqa: BLE001 - capture is best-effort
                    capture_status["error"] = f"{type(exc).__name__}: {exc}"
            _record_aot_status(name, capture_status)
            _log_aot_status(name, capture_status)
            return compiled(
                carry,
                namelist,
                start,
                clock_base,
                n_steps=int(n_steps),
                cadence=int(cadence),
            )
        except BaseException as exc:  # noqa: BLE001 - preserve legacy fallback
            status = {
                "name": name,
                "loaded": False,
                "source": "fallback:jit-compile-exception",
                "error": f"{type(exc).__name__}: {exc}",
                "hlo_sha256": hlo_sha256,
                "cheap_key": cheap_key,
            }
            _record_aot_status(name, status)
            _log_aot_status(name, status)
            return _advance_chunk(
                carry,
                namelist,
                start,
                clock_base,
                n_steps=int(n_steps),
                cadence=cadence,
            )

    def advance(name: str, carry: OperationalCarry, start_step: int, n_steps: int) -> OperationalCarry:
        namelist = tree.domains[name].namelist
        cadence = int(namelist.radiation_cadence_steps)
        if cadence <= 0:
            raise ValueError(f"{name}: radiation_cadence_steps must be positive")
        start = jnp.asarray(int(start_step), dtype=jnp.int32)
        clock_base = _clock_bases[name]

        # Step C (vNext cheap-key manifest): locate the serialized executable for
        # THIS exact per-domain/per-shape variant via a metadata-only cheap_key --
        # NO lowering on the warm path. One domain may see multiple carry-shape
        # variants during nested coupling segments, so the memo keys on
        # (domain, cheap_key). The first cold miss lowers + compiles + serializes
        # that variant (under both the cheap_key and the hlo address); every later
        # warm process computes the cheap_key in microseconds and loads the blob.
        # Any failure remains fail-open to the jitted path -- never wrong, slower.
        if aot_on:
            if not _aot_fori_mode_enabled():
                status = {
                    "name": name,
                    "loaded": False,
                    "source": "fallback:loop-mode",
                    "error": "AOT Step C supports only the fori-loop advance mode",
                    "hlo_sha256": None,
                }
                _record_aot_status(name, status)
                _log_aot_status(name, status)
            else:
                # --- WARM PATH: metadata-only cheap_key, no lowering. ---
                ckey = _cheap_key_for(
                    carry,
                    namelist,
                    start,
                    clock_base,
                    n_steps=int(n_steps),
                    cadence=int(cadence),
                )
                if ckey is not None:
                    ck_state = (name, ckey)
                    aot_call, ck_status = _aot_advance_by_cheap_key(name, ckey)
                    if aot_call is not None and ck_state not in _aot_verify_failed:
                        # Verify-mode: confirm the blob lowers to its recorded HLO
                        # ONCE per key, fail-CLOSED on a mismatch (a missed cheap_key
                        # determinant becomes a loud fallback, not a silent wrong).
                        verified_ok = True
                        if aot_verify and ck_state not in _aot_verified:
                            verified_ok = _verify_cheap_key_blob(
                                name,
                                ckey,
                                ck_status.get("meta_hlo_sha256"),
                                carry,
                                namelist,
                                start,
                                clock_base,
                                n_steps=int(n_steps),
                                cadence=int(cadence),
                            )
                        if verified_ok:
                            ck_status["source"] = "aot_blob"
                            ck_status["loaded"] = True
                            _record_aot_status(name, ck_status)
                            _log_aot_status(name, ck_status)
                            try:
                                return aot_call(
                                    carry,
                                    namelist,
                                    start,
                                    clock_base,
                                    n_steps=int(n_steps),
                                    cadence=int(cadence),
                                )
                            except BaseException as exc:  # noqa: BLE001 - fail-open
                                _aot_calls[ck_state] = None
                                status = {
                                    "name": name,
                                    "loaded": False,
                                    "source": "fallback:jit(exec-error)",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "cheap_key": ckey,
                                }
                                _record_aot_status(name, status)
                                _log_aot_status(name, status)
                        else:
                            # Verify FAILED: never load this key again; compile fresh.
                            _aot_calls[ck_state] = None
                    else:
                        _record_aot_status(name, ck_status)
                        _log_aot_status(name, ck_status)
                    # MISS / exec-error / verify-fail: lower + compile + serialize
                    # the exact variant (also writes the cheap-key-addressed blob).
                    lowered, hlo_sha256, lower_error = _lower_advance_variant(
                        carry,
                        namelist,
                        start,
                        clock_base,
                        n_steps=int(n_steps),
                        cadence=int(cadence),
                    )
                    if lower_error:
                        status = {
                            "name": name,
                            "loaded": False,
                            "source": "fallback:lower-error",
                            "error": lower_error,
                            "hlo_sha256": hlo_sha256,
                            "cheap_key": ckey,
                        }
                        _record_aot_status(name, status)
                        _log_aot_status(name, status)
                    else:
                        return _compile_capture_and_call(
                            name,
                            lowered,
                            hlo_sha256,
                            carry,
                            namelist,
                            start,
                            clock_base,
                            n_steps=int(n_steps),
                            cadence=int(cadence),
                            cheap_key=ckey,
                        )
                    # lower-error -> drop through to the plain jitted path below.
                    return _advance_chunk(
                        carry,
                        namelist,
                        start,
                        clock_base,
                        n_steps=int(n_steps),
                        cadence=cadence,
                    )

                # --- cheap_key unavailable: legacy lower-only fingerprint path. ---
                lowered, hlo_sha256, lower_error = _lower_advance_variant(
                    carry,
                    namelist,
                    start,
                    clock_base,
                    n_steps=int(n_steps),
                    cadence=int(cadence),
                )
                if lower_error:
                    status = {
                        "name": name,
                        "loaded": False,
                        "source": "fallback:lower-error",
                        "error": lower_error,
                        "hlo_sha256": hlo_sha256,
                    }
                    _record_aot_status(name, status)
                    _log_aot_status(name, status)
                elif hlo_sha256:
                    aot_call = _aot_advance_for(name, hlo_sha256)
                    if aot_call is not None:
                        try:
                            return aot_call(
                                carry,
                                namelist,
                                start,
                                clock_base,
                                n_steps=int(n_steps),
                                cadence=int(cadence),
                            )
                        except BaseException as exc:  # noqa: BLE001 - fail-open
                            _aot_calls[(name, hlo_sha256)] = None
                            status = {
                                "name": name,
                                "loaded": False,
                                "source": "fallback:jit(exec-error)",
                                "error": f"{type(exc).__name__}: {exc}",
                                "hlo_sha256": hlo_sha256,
                            }
                            _record_aot_status(name, status)
                            _log_aot_status(name, status)
                    return _compile_capture_and_call(
                        name,
                        lowered,
                        hlo_sha256,
                        carry,
                        namelist,
                        start,
                        clock_base,
                        n_steps=int(n_steps),
                        cadence=int(cadence),
                    )
                else:
                    status = {
                        "name": name,
                        "loaded": False,
                        "source": "fallback:no-hlo-digest",
                        "error": "lowered HLO digest unavailable",
                        "hlo_sha256": None,
                    }
                    _record_aot_status(name, status)
                    _log_aot_status(name, status)

        return _advance_chunk(
            carry,
            namelist,
            start,
            clock_base,
            n_steps=int(n_steps),
            cadence=cadence,
        )

    return advance


def nested_aot_report() -> dict[str, object]:
    """Snapshot of the last AOT-load outcome per domain (Step C diagnostics)."""
    domains = NESTED_AOT_STATUS.get("domains") or {}
    return {
        "enabled": bool(NESTED_AOT_STATUS.get("enabled")),
        "verify": bool(NESTED_AOT_STATUS.get("verify")),
        "domains": dict(domains) if isinstance(domains, dict) else {},
    }


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


@dataclass(frozen=True)
class _FusedAuxNamelist:
    """Namelist-like static carrier for the fused-cascade cheap-key.

    ``aot_cheap_key.static_config_hash`` hashes ``tree_flatten()[1]``. The fused
    jit closes over parent/child namelists and edge geometry rather than taking a
    single ``OperationalNamelist`` argument, so this synthetic object exposes the
    complete closure determinant set through the same static-aux channel.
    """

    aux: Any

    def tree_flatten(self):
        return (), self.aux


def _namelist_static_aux(namelist: Any) -> Any:
    flatten = getattr(namelist, "tree_flatten", None)
    if callable(flatten):
        _children, aux = flatten()
        return aux
    attrs: dict[str, Any] = {}
    for cls in reversed(type(namelist).__mro__):
        for name, value in vars(cls).items():
            if not name.startswith("_") and not callable(value):
                attrs[name] = value
    if hasattr(namelist, "__dict__"):
        attrs.update(
            (name, value)
            for name, value in vars(namelist).items()
            if not name.startswith("_")
        )
    return (type(namelist).__module__, type(namelist).__qualname__, tuple(sorted(attrs.items())))


def _fused_source_fingerprint() -> str:
    """Conservative source digest for fused-only orchestration callees."""
    try:
        from gpuwrf.runtime import aot_cheap_key as _ck

        files = [
            Path(__file__).resolve(),
            Path(build_child_boundary_package.__code__.co_filename).resolve(),
        ]
        payload = []
        for path in files:
            data = path.read_bytes()
            payload.append((path.name, len(data), _ck.canonical_digest(data)))
        return _ck.canonical_digest(("fused-source", tuple(payload)))
    except Exception:  # noqa: BLE001 - fail-open to a stable marker
        return "fused-source-unavailable"


def _build_fused_aux_namelist(
    *,
    parent_namelist: OperationalNamelist,
    parent_cadence: int,
    child_names: tuple[str, ...],
    child_namelists: tuple[OperationalNamelist, ...],
    child_weights: tuple[NestForceWeights, ...],
    child_bdy_widths: tuple[int, ...],
    child_ratios: tuple[int, ...],
    child_cadences: tuple[int, ...],
) -> _FusedAuxNamelist:
    """Build the synthetic static aux that fully identifies a fused cascade."""
    child_aux = []
    for name, namelist, weights, bdy_width, ratio, cadence in zip(
        child_names,
        child_namelists,
        child_weights,
        child_bdy_widths,
        child_ratios,
        child_cadences,
        strict=True,
    ):
        child_aux.append(
            {
                "name": str(name),
                "namelist_static_aux": _namelist_static_aux(namelist),
                "weights": weights,
                "bdy_width": int(bdy_width),
                "ratio": int(ratio),
                "cadence": int(cadence),
            }
        )
    return _FusedAuxNamelist(
        (
            "gpuwrf-fused-cascade-aot-v1",
            {
                "parent_namelist_static_aux": _namelist_static_aux(parent_namelist),
                "parent_cadence": int(parent_cadence),
                "child_count": len(child_aux),
                "children": tuple(child_aux),
                "fused_source": _fused_source_fingerprint(),
            },
        )
    )


def _fused_cascade_cheap_key(
    fused_jit: Any,
    fused_aux: _FusedAuxNamelist,
    parent_carry: Any,
    child_carries: tuple[Any, ...],
    parent_start: int,
    child_starts: tuple[int, ...],
    parent_clock_base: Any,
    child_clock_bases: tuple[Any, ...],
) -> str | None:
    """Metadata-only AOT key for a fused cascade call; ``None`` means fail-open."""
    try:
        from gpuwrf.runtime import aot_cheap_key as _ck

        return _ck.cheap_key(
            fused_jit,
            (
                parent_carry,
                child_carries,
                int(parent_start),
                tuple(int(s) for s in child_starts),
                parent_clock_base,
                child_clock_bases,
            ),
            {},
            fused_aux,
        )
    except BaseException:  # noqa: BLE001 - no key -> normal fused jit
        return None


def _record_nested_aot_status(name: str, status: dict[str, Any]) -> None:
    try:
        domains = NESTED_AOT_STATUS.setdefault("domains", {})
        if isinstance(domains, dict):
            domains[name] = dict(status)
    except Exception:  # noqa: BLE001
        pass


def _log_nested_aot_status(name: str, status: dict[str, Any]) -> None:
    """Emit one stderr line per AOT load/fallback for GPU gate visibility."""
    try:
        loaded = bool(status.get("loaded"))
        source = status.get("source") or ("aot_blob" if loaded else "fallback:jit")
        parts = [
            f"[gpuwrf:nested-aot] domain={name}",
            f"loaded={str(loaded).lower()}",
            f"source={source}",
        ]
        blob_path = status.get("blob_path")
        if blob_path:
            parts.append(f"path={blob_path}")
        hlo = status.get("hlo_sha256") or status.get("meta_hlo_sha256")
        if hlo:
            parts.append(f"hlo={str(hlo)[:12]}")
        ckey = status.get("cheap_key")
        if ckey:
            parts.append(f"cheap_key={str(ckey)[:12]}")
        error = status.get("error")
        if error:
            parts.append(f"error={str(error).replace(chr(10), ' | ')}")
        sys.stderr.write(" ".join(parts) + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _aval_signature(args: tuple[Any, ...]) -> tuple[tuple[tuple[int, ...], str, bool], ...]:
    """Cheap runtime-arg aval signature for cached compiled executable reuse."""

    sig: list[tuple[tuple[int, ...], str, bool]] = []
    for leaf in jax.tree_util.tree_leaves(args):
        try:
            aval = jax.typeof(leaf)
            shape = tuple(int(v) for v in getattr(aval, "shape", ()))
            dtype = str(getattr(aval, "dtype", type(leaf).__name__))
            weak_type = bool(getattr(aval, "weak_type", False))
        except Exception:  # noqa: BLE001 - only a cache key; fail closed to unique-ish type
            raw_shape = getattr(leaf, "shape", ())
            shape = tuple(int(v) for v in raw_shape)
            dtype = str(getattr(leaf, "dtype", type(leaf).__name__))
            weak_type = False
        sig.append((shape, dtype, weak_type))
    return tuple(sig)


def _build_fused_cascade_program(
    *,
    parent_name: str,
    parent_namelist: OperationalNamelist,
    parent_cadence: int,
    child_names: tuple[str, ...],
    child_namelists: tuple[OperationalNamelist, ...],
    child_weights: tuple[NestForceWeights, ...],
    child_bdy_widths: tuple[int, ...],
    child_ratios: tuple[int, ...],
    child_cadences: tuple[int, ...],
) -> FusedSubstepFn:
    """Build one jitted parent-substep cascade for a flat one-way leaf subtree."""

    parent_cad = int(parent_cadence)
    n_children = len(child_namelists)
    aot_name = f"fused/{parent_name}"
    aot_on = _nested_aot_enabled()
    aot_verify = _nested_aot_verify_enabled()
    NESTED_AOT_STATUS["enabled"] = bool(aot_on)
    NESTED_AOT_STATUS["verify"] = bool(aot_verify)
    fused_aux = _build_fused_aux_namelist(
        parent_namelist=parent_namelist,
        parent_cadence=parent_cadence,
        child_names=tuple(str(name) for name in child_names),
        child_namelists=child_namelists,
        child_weights=child_weights,
        child_bdy_widths=child_bdy_widths,
        child_ratios=child_ratios,
        child_cadences=child_cadences,
    )
    # #91: per-run date scalars built ONCE on the host and passed into the jitted
    # cascade as TRACED arguments (parent_cb / child_cbs) so the fused HLO is
    # date-independent (cross-date compile-cache hit; numerically inert).
    parent_clock_base = build_clock_base(parent_namelist)
    child_clock_bases = tuple(build_clock_base(nl) for nl in child_namelists)

    @jax.jit
    def fused_jit(parent_carry, child_carries, parent_start, child_starts, parent_cb, child_cbs):
        parent_new = _advance_chunk(
            parent_carry,
            parent_namelist,
            jnp.asarray(parent_start, dtype=jnp.int32),
            parent_cb,
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
                child_cbs[idx],
                n_steps=int(child_ratios[idx]),
                cadence=int(child_cadences[idx]),
            )
            new_children.append(child_new)
        return parent_new, tuple(new_children)

    cached_calls: dict[tuple[tuple[tuple[int, ...], str, bool], ...], tuple[Any, str | None]] = {}
    max_cached_calls = 16
    verified_keys: set[str] = set()
    verify_failed_keys: set[str] = set()

    def _args(parent_carry, child_carries, parent_start, child_starts):
        return (
            parent_carry,
            tuple(child_carries),
            int(parent_start),
            tuple(int(s) for s in child_starts),
            parent_clock_base,
            child_clock_bases,
        )

    def _lower_fused(args: tuple[Any, ...]) -> tuple[Any | None, str | None, str | None]:
        try:
            from gpuwrf.runtime import aot_executable as _aotx

            lowered = fused_jit.lower(*args)
            hlo = _aotx.hlo_sha256_from_lowered(lowered)
            if not hlo:
                return lowered, None, "lowered HLO digest unavailable"
            return lowered, hlo, None
        except BaseException as exc:  # noqa: BLE001 - fail-open to normal fused jit
            return None, None, f"{type(exc).__name__}: {exc}"

    def _store_cached_call(
        sig: tuple[tuple[tuple[int, ...], str, bool], ...],
        call: Any,
        ckey: str | None,
    ) -> None:
        if sig not in cached_calls and len(cached_calls) >= max_cached_calls:
            cached_calls.pop(next(iter(cached_calls)), None)
        cached_calls[sig] = (call, ckey)

    def _compile_capture_and_call(
        args: tuple[Any, ...],
        ckey: str | None,
        sig: tuple[tuple[tuple[int, ...], str, bool], ...],
    ):
        lowered, hlo_sha256, lower_error = _lower_fused(args)
        if lower_error or lowered is None:
            status = {
                "name": aot_name,
                "loaded": False,
                "source": "fallback:fused-lower-error",
                "error": lower_error,
                "cheap_key": ckey,
            }
            _record_nested_aot_status(aot_name, status)
            _log_nested_aot_status(aot_name, status)
            return fused_jit(*args)
        try:
            compiled = lowered.compile()
            _store_cached_call(sig, compiled, ckey)
            status: dict[str, Any] = {
                "name": aot_name,
                "loaded": False,
                "source": "fallback:fused-jit-compiled",
                "error": None,
                "hlo_sha256": hlo_sha256,
                "cheap_key": ckey,
            }
            if ckey:
                try:
                    from gpuwrf.runtime import aot_cheap_key as _ck
                    from gpuwrf.runtime import aot_precompile as _aotp

                    ser = _aotp._serialize_domain_blob(
                        aot_name,
                        compiled,
                        None,
                        hlo_sha256=hlo_sha256,
                        lowered=lowered,
                        cheap_key=ckey,
                        key_schema=_ck.KEY_SCHEMA,
                    )
                    status.update(ser)
                    if ser.get("aot_written"):
                        status["source"] = "fallback:fused-jit-compiled+aot-captured"
                    elif ser.get("aot_error"):
                        status["error"] = ser.get("aot_error")
                except BaseException as exc:  # noqa: BLE001 - capture is best-effort
                    status["error"] = f"{type(exc).__name__}: {exc}"
            _record_nested_aot_status(aot_name, status)
            _log_nested_aot_status(aot_name, status)
            return compiled(*args)
        except BaseException as exc:  # noqa: BLE001 - preserve normal fused fallback
            cached_calls.pop(sig, None)
            status = {
                "name": aot_name,
                "loaded": False,
                "source": "fallback:fused-jit-compile-exception",
                "error": f"{type(exc).__name__}: {exc}",
                "hlo_sha256": hlo_sha256,
                "cheap_key": ckey,
            }
            _record_nested_aot_status(aot_name, status)
            _log_nested_aot_status(aot_name, status)
            return fused_jit(*args)

    def _verify_loaded_blob(args: tuple[Any, ...], ckey: str, meta_hlo: str | None) -> bool:
        if ckey in verified_keys:
            return True
        if ckey in verify_failed_keys:
            return False
        _lowered, hlo_now, lower_error = _lower_fused(args)
        if lower_error or not hlo_now or not meta_hlo:
            verify_failed_keys.add(ckey)
            status = {
                "name": aot_name,
                "loaded": False,
                "source": "fallback:fused-verify-error",
                "error": lower_error or "missing HLO digest for verify",
                "cheap_key": ckey,
            }
            _record_nested_aot_status(aot_name, status)
            _log_nested_aot_status(aot_name, status)
            return False
        if hlo_now != meta_hlo:
            verify_failed_keys.add(ckey)
            try:
                from gpuwrf.runtime import aot_precompile as _aotp

                _aotp.quarantine_cheap_key(
                    aot_name,
                    ckey,
                    reason="fused verify-mode HLO mismatch",
                    detail={"meta_hlo": str(meta_hlo), "live_hlo": str(hlo_now)},
                )
            except Exception:  # noqa: BLE001 - quarantine is best-effort
                pass
            status = {
                "name": aot_name,
                "loaded": False,
                "source": "fallback:fused-verify-MISMATCH",
                "error": (
                    "AOT VERIFY FAILED for fused cascade: blob meta hlo="
                    f"{str(meta_hlo)[:12]} != live lowered hlo={str(hlo_now)[:12]}"
                ),
                "cheap_key": ckey,
            }
            _record_nested_aot_status(aot_name, status)
            _log_nested_aot_status(aot_name, status)
            return False
        verified_keys.add(ckey)
        return True

    def fused(parent_carry, child_carries, parent_start, child_starts):
        # The traced clock bases ride as runtime jit arguments (NOT closed-over
        # constants), keeping the compiled cascade identical for every date.
        args = _args(parent_carry, child_carries, parent_start, child_starts)
        if not aot_on:
            return fused_jit(*args)

        sig = _aval_signature(args)
        cached = cached_calls.get(sig)
        if cached is not None:
            cached_call, cached_key = cached
            try:
                return cached_call(*args)
            except BaseException as exc:  # noqa: BLE001 - shape drift or execute error
                status = {
                    "name": aot_name,
                    "loaded": False,
                    "source": "fallback:fused-cached-call-error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "cheap_key": cached_key,
                }
                _record_nested_aot_status(aot_name, status)
                _log_nested_aot_status(aot_name, status)
                cached_calls.pop(sig, None)

        ckey = _fused_cascade_cheap_key(
            fused_jit,
            fused_aux,
            parent_carry,
            tuple(child_carries),
            int(parent_start),
            tuple(int(s) for s in child_starts),
            parent_clock_base,
            child_clock_bases,
        )
        if ckey is None:
            return fused_jit(*args)

        try:
            from gpuwrf.runtime import aot_precompile as _aotp

            loaded = _aotp.load_domain_blob(aot_name, cheap_key=ckey, return_status=True)
            if isinstance(loaded, tuple) and len(loaded) == 2:
                call, status = loaded
                if isinstance(status, dict):
                    if status.get("hlo_sha256") is None and status.get("meta_hlo_sha256"):
                        status["hlo_sha256"] = status["meta_hlo_sha256"]
                    _record_nested_aot_status(aot_name, status)
                    _log_nested_aot_status(aot_name, status)
            else:
                call = loaded
                status = {
                    "name": aot_name,
                    "loaded": call is not None,
                    "source": "aot_blob" if call is not None else "fallback:fused-jit",
                    "cheap_key": ckey,
                }
                _record_nested_aot_status(aot_name, status)
                _log_nested_aot_status(aot_name, status)
            if call is not None and (
                not aot_verify
                or _verify_loaded_blob(args, ckey, status.get("meta_hlo_sha256"))
            ):
                try:
                    out = call(*args)
                    _store_cached_call(sig, call, ckey)
                    return out
                except BaseException as exc:  # noqa: BLE001 - fail-open to compile
                    status = {
                        "name": aot_name,
                        "loaded": False,
                        "source": "fallback:fused-aot-exec-error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "cheap_key": ckey,
                    }
                    _record_nested_aot_status(aot_name, status)
                    _log_nested_aot_status(aot_name, status)
        except BaseException as exc:  # noqa: BLE001 - fail-open to compile
            status = {
                "name": aot_name,
                "loaded": False,
                "source": "fallback:fused-load-exception",
                "error": f"{type(exc).__name__}: {exc}",
                "cheap_key": ckey,
            }
            _record_nested_aot_status(aot_name, status)
            _log_nested_aot_status(aot_name, status)

        return _compile_capture_and_call(args, ckey, sig)

    return fused


_FUSED_PROGRAM_CACHE: dict[int, "tuple[weakref.ref[Any], dict[str, FusedSubstepFn]]"] = {}
_FALSEY_ENV = {"0", "false", "off", "no", ""}
_TRUEY_ENV = {"1", "true", "on", "yes"}


def _env_flag(name: str) -> str | None:
    value = os.environ.get(name)
    return None if value is None else value.strip().lower()


# B2 de-fuse knob status (observability for tests / cache_report). Records why
# the eager per-domain compile path was (or was not) selected on the last call to
# :func:`_nested_fuse_default_enabled`. Numerically inert -- both paths are
# bit-identical (see the eager opt-out that already exists for identity proofs).
NESTED_DEFUSE_STATUS: dict[str, object] = {
    "defused": False,
    "source": None,
}


def _nested_defuse_for_compile_ram() -> bool:
    """Whether to compile the nest PER-DOMAIN instead of one fused module (B2).

    OPT-IN, default OFF. ``GPUWRF_NESTED_DEFUSE_COMPILE`` truthy
    (``1``/``true``/``on``/``yes``) selects the eager per-domain advance path,
    which compiles each domain's ``_advance_chunk`` as its OWN executable instead
    of fusing the parent + all its leaves into one ``jax.jit`` cascade program.

    WHY (compile-RAM, NOT numerics): a fused cascade lowers the parent advance and
    every child advance into a SINGLE XLA module, so XLA holds the whole tower's
    HLO + buffer-assignment live at once -- peak COMPILE-RAM scales with the fused
    leaf count (the ~K-leaf nest is the worst case; this is what stressed the paid
    B200 cold compile). The eager per-domain path compiles each domain
    independently, so peak compile-RAM is ~that of the single LARGEST domain (≈ K×
    lower for a K-leaf tower). The trade is runtime throughput (the fused cascade
    keeps the GPU queue fuller), so this is an explicit low-host-RAM opt-in rather
    than the runtime default. AOT cheap-key loading works on both the fused runtime
    path and this de-fused fallback.

    NUMERICALLY INERT: the eager per-domain path is the SAME path already used by
    the ``GPUWRF_BITWISE`` / ``GPUWRF_NESTED_FUSE=0`` identity opt-outs (the fused
    cascade is a fused-but-bit-identical rewrite of it), so de-fusing changes no
    dispatched op, cadence, or float result -- only how XLA partitions the
    compile. Distinct from those vars by INTENT: this one is a compile-RAM lever,
    not an identity/debug switch, and is logged as such.
    """
    flag = _env_flag("GPUWRF_NESTED_DEFUSE_COMPILE")
    return flag is not None and flag in _TRUEY_ENV


def _nested_fuse_default_enabled() -> bool:
    """Default-FUSED nest runtime, with explicit eager/de-fuse opt-outs.

    The fused cascade is the runtime default because it is the measured
    higher-throughput executable. ``GPUWRF_NESTED_DEFUSE_COMPILE=1`` remains an
    explicit low-host-compile-RAM lever, and ``GPUWRF_NESTED_FUSE=0`` /
    ``GPUWRF_BITWISE=1`` keep the eager per-domain identity/debug path.
    """

    # B2 compile-RAM de-fuse opt-in: force the eager per-domain compile path. Kept
    # FIRST so it is honoured regardless of the identity vars below; recorded in
    # NESTED_DEFUSE_STATUS for observability. Numerically inert (same eager path).
    if _nested_defuse_for_compile_ram():
        NESTED_DEFUSE_STATUS["defused"] = True
        NESTED_DEFUSE_STATUS["source"] = "env:GPUWRF_NESTED_DEFUSE_COMPILE"
        return False

    bitwise = _env_flag("GPUWRF_BITWISE")
    if bitwise is not None and bitwise not in _FALSEY_ENV:
        NESTED_DEFUSE_STATUS["defused"] = True
        NESTED_DEFUSE_STATUS["source"] = "env:GPUWRF_BITWISE"
        return False
    fuse = _env_flag("GPUWRF_NESTED_FUSE")
    if fuse is None:
        NESTED_DEFUSE_STATUS["defused"] = False
        NESTED_DEFUSE_STATUS["source"] = "default-fused"
        return True
    if fuse in _FALSEY_ENV:
        NESTED_DEFUSE_STATUS["defused"] = True
        NESTED_DEFUSE_STATUS["source"] = "env:GPUWRF_NESTED_FUSE=0"
        return False
    if fuse in _TRUEY_ENV:
        NESTED_DEFUSE_STATUS["defused"] = False
        NESTED_DEFUSE_STATUS["source"] = "env:GPUWRF_NESTED_FUSE=1"
        return True
    # Malformed GPUWRF_NESTED_FUSE value: fall back to the fused runtime default.
    NESTED_DEFUSE_STATUS["defused"] = False
    NESTED_DEFUSE_STATUS["source"] = "default-fused"
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
                parent_name=str(parent_name),
                parent_namelist=parent_namelist,
                parent_cadence=parent_cadence,
                child_names=tuple(edge.child for edge in edges),
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
    max_event_tail: int | None = None,
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
        max_event_tail=max_event_tail,
    )


__all__ = [
    "DomainBundle",
    "DomainEdge",
    "DomainTree",
    "DomainTreeResult",
    "NESTED_AOT_STATUS",
    "NESTED_DEFUSE_STATUS",
    "NESTED_PRECOMPILE_STATUS",
    "build_live_nested_boundary_config",
    "maybe_prewarm_defused_nest",
    "nested_aot_report",
    "nested_defuse_report",
    "nested_precompile_report",
    "run_domain_tree_callbacks",
    "run_operational_domain_tree",
    "with_live_child_boundary_config",
]


# vNext: cross-domain parallel-compile status (observability for the manager's
# GPU A/B + tests). Records the outcome of the last
# :func:`maybe_prewarm_defused_nest` call. Numerically inert -- prewarm only
# POPULATES the shared cache that the unchanged eager loop warm-hits.
NESTED_PRECOMPILE_STATUS: dict[str, object] = {
    "attempted": False,
    "active": False,
    "source": None,
    "workers": None,
    "verify": None,
    "report": None,
    "error": None,
}


def _nested_parallel_compile_workers() -> int | None:
    """Parse ``GPUWRF_NESTED_PARALLEL_COMPILE``.

    Returns:
      * ``0``    -> explicit opt-OUT (caller skips the parallel prewarm);
      * ``N>0``  -> explicit worker-count override;
      * ``None`` -> unset/auto (caller uses the RAM/CPU-derived default).
    """
    raw = os.environ.get("GPUWRF_NESTED_PARALLEL_COMPILE")
    if raw is None or raw.strip() == "":
        return None
    try:
        val = int(raw.strip())
    except ValueError:
        return None
    return max(0, val)


def _nested_parallel_verify_enabled() -> bool:
    """Parse ``GPUWRF_NESTED_PARALLEL_VERIFY`` (default OFF).

    Truthy values ``1/true/yes/on`` (case-insensitive) enable the parent-side
    exact-key warm-hit verification pass in :func:`prewarm_defused_nest`. That
    pass re-lowers all N huge ``_advance_chunk_fori`` modules sequentially (the
    ~30 min re-lowering cost) and is therefore NET-NEGATIVE for wall-clock; it is
    a DIAGNOSTIC only, so it is OFF by default. Anything else (unset, ``0``,
    empty, junk) leaves it OFF."""
    raw = os.environ.get("GPUWRF_NESTED_PARALLEL_VERIFY")
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def maybe_prewarm_defused_nest(
    tree: "DomainTree",
    *,
    carries: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Gate + fire the cross-domain PARALLEL pre-compile for the de-fuse path.

    No-op UNLESS the de-fuse compile path is active (``not
    _nested_fuse_default_enabled()`` -- i.e. ``GPUWRF_NESTED_DEFUSE_COMPILE=1`` /
    ``GPUWRF_NESTED_FUSE=0`` / ``GPUWRF_BITWISE``), because only then are the N
    independent per-domain modules compiled, which is what parallelizes. The
    fused DEFAULT compiles ONE module, so there is nothing to fan out.

    Honors ``GPUWRF_NESTED_PARALLEL_COMPILE`` (=0 opt-out; =N worker override).
    Calls :func:`aot_precompile.prewarm_defused_nest` to warm the shared
    version-keyed cache, then records the report. ``carries`` should be the exact
    runtime carry map the eager integration loop will use, after post-init scheme
    seeding and device commit; this keeps the child cache key identical to the
    parent eager key. FAILS OPEN: any exception is captured in
    :data:`NESTED_PRECOMPILE_STATUS` and the function returns, so the integration
    loop just cold-compiles sequentially as today (never wrong, never
    non-bit-identical -- only loses the speedup). Numerically inert.

    Returns a copy of :data:`NESTED_PRECOMPILE_STATUS`.
    """
    NESTED_PRECOMPILE_STATUS.update(
        {
            "attempted": True,
            "active": False,
            "source": None,
            "workers": None,
            "verify": None,
            "report": None,
            "error": None,
        }
    )
    try:
        # (a) only when de-fused.
        if _nested_fuse_default_enabled():
            NESTED_PRECOMPILE_STATUS["source"] = "skip:fused-default"
            return dict(NESTED_PRECOMPILE_STATUS)

        # (b) honor the opt-out / worker override.
        #
        # SPAWN-SAFETY: the parallel prewarm spawns
        # child processes (multiprocessing "spawn" re-imports the entry module),
        # which corrupts/recurses an UNGUARDED entry point (pytest, web UI,
        # ad-hoc scripts with no ``if __name__ == "__main__"`` guard). Even when
        # de-fuse is explicitly selected, the spawning parallel prewarm stays
        # strictly OPT-IN: fire ONLY when GPUWRF_NESTED_PARALLEL_COMPILE is
        # EXPLICITLY set to N>0. When it is unset (workers is None) we take the
        # SEQUENTIAL no-spawn de-fuse sub-mode. =0 is the explicit opt-out.
        # Numerically inert.
        workers = _nested_parallel_compile_workers()
        if workers == 0:
            NESTED_PRECOMPILE_STATUS["source"] = "skip:GPUWRF_NESTED_PARALLEL_COMPILE=0"
            return dict(NESTED_PRECOMPILE_STATUS)
        if workers is None:
            # Unset => sequential de-fuse sub-mode (no spawn). Parallel prewarm
            # is opt-in via GPUWRF_NESTED_PARALLEL_COMPILE=N.
            NESTED_PRECOMPILE_STATUS["source"] = "skip:defuse-sequential-no-parallel"
            return dict(NESTED_PRECOMPILE_STATUS)

        # (c) fire the parallel prewarm (its own driver is also fail-open). The
        # parent-side warm-hit verification re-lowers all N modules (~30 min,
        # NET-NEGATIVE for wall) and is OFF unless GPUWRF_NESTED_PARALLEL_VERIFY
        # is explicitly truthy -- the eager loop warm-hits anyway.
        from gpuwrf.runtime import aot_precompile

        verify = _nested_parallel_verify_enabled()
        report = aot_precompile.prewarm_defused_nest(
            tree, carries=carries, max_workers=workers, verify=verify
        )
        NESTED_PRECOMPILE_STATUS["verify"] = verify
        NESTED_PRECOMPILE_STATUS["active"] = True
        # workers is now always an explicit N>0 here (unset takes the sequential
        # no-spawn sub-mode above), so the source is always the explicit override.
        NESTED_PRECOMPILE_STATUS["source"] = f"GPUWRF_NESTED_PARALLEL_COMPILE={workers}"
        NESTED_PRECOMPILE_STATUS["workers"] = report.get("workers")
        NESTED_PRECOMPILE_STATUS["report"] = report
        if report.get("error"):
            NESTED_PRECOMPILE_STATUS["error"] = report.get("error")
    except BaseException as exc:  # noqa: BLE001 - the whole gate is fail-open
        NESTED_PRECOMPILE_STATUS["error"] = f"{type(exc).__name__}: {exc}"
    return dict(NESTED_PRECOMPILE_STATUS)


def nested_precompile_report() -> dict[str, object]:
    """Snapshot of the last cross-domain parallel-compile attempt (vNext)."""
    return dict(NESTED_PRECOMPILE_STATUS)


def nested_defuse_report() -> dict[str, object]:
    """Snapshot of the nest fuse/de-fuse compile decision (B2).

    Recomputes :func:`_nested_fuse_default_enabled` so the report reflects the
    CURRENT environment (the env vars are read live), then returns a copy of
    :data:`NESTED_DEFUSE_STATUS` plus the effective ``fused`` boolean. Pure /
    side-effect-free apart from refreshing the status dict; never raises.
    """
    fused = _nested_fuse_default_enabled()
    return {
        "fused": fused,
        "defused": bool(NESTED_DEFUSE_STATUS.get("defused")),
        "source": NESTED_DEFUSE_STATUS.get("source"),
        "env_help": nested_defuse_env_help(),
    }


def nested_defuse_env_help() -> str:
    """One-line human summary of the nest de-fuse compile knob (B2)."""
    return (
        "Nest compile-fusion env vars: the DEFAULT is FUSED + AOT "
        "(one fused cascade executable for safe flat leaf subtrees, higher runtime "
        "throughput, warm-start by fused cheap-key executable load). "
        "GPUWRF_NESTED_DEFUSE_COMPILE=1 / GPUWRF_NESTED_FUSE=0 / GPUWRF_BITWISE also "
        "select the eager per-domain path -- useful as a low-host-compile-RAM or "
        "bitwise/debug fallback, with documented runtime cost. "
        "GPUWRF_NESTED_PARALLEL_COMPILE controls the cross-domain PARALLEL "
        "pre-compile of the de-fuse path (=0 opt-out, =N worker count): when SET to "
        "N>0 it compiles the N independent per-domain modules CONCURRENTLY in "
        "SPAWNED processes to cut cold wall from Sum(N) toward max(one body). It is "
        "OPT-IN: when UNSET the explicit de-fuse path stays SEQUENTIAL (no spawn) "
        "so unguarded entry points (pytest/web/scripts) are spawn-safe. "
        "Numerically inert (warms the shared cache the unchanged eager loop hits). "
        "Requires the de-fuse path active. "
        "GPUWRF_NESTED_PARALLEL_VERIFY=1 (default OFF) re-enables the diagnostic "
        "parent-side exact-key warm-hit verification, which re-lowers all N "
        "modules sequentially (~30 min, NET-NEGATIVE for wall); off by default "
        "since the eager loop warm-hits each domain anyway. "
        "GPUWRF_NESTED_AOT (DEFAULT-ON; set =0 to disable) enables AOT executable "
        "serialization + the cheap-key manifest: the fused runtime serializes "
        "fused/<parent> cascade executables under an edge-geometry-aware cheap key, "
        "and the de-fuse path serializes each domain's _advance_chunk_fori under "
        "<cache>/aot/<version_tag>/<domain>/k_<cheap_key>.xlaexec. Warm processes "
        "compute metadata-only cheap keys (version+fn+static aux+carry-avals+"
        "trace-env, NO lowering) to locate + DESERIALIZE + execute the matching "
        "blob in seconds. Identity-preserving (the blob IS the cached executable), "
        "version-fingerprint-guarded, and fail-open (cheap_key miss / fingerprint "
        "mismatch / blob-integrity / corrupt -> compile+serialize, never wrong). "
        "GPUWRF_AOT_VERIFY=1 (default OFF) lowers ONCE per (domain,cheap_key) per "
        "process and asserts the loaded blob's HLO matches its recorded digest, "
        "failing CLOSED (loud fallback + fresh compile) on any mismatch -- run it "
        "for the first warm-start of each new version, then off in production."
    )
