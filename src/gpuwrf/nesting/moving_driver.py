"""Operational moving-nest driver (WRF ``med_nest_move`` faithful, v0.23 G2).

This module turns the v0.22 moving-nest primitives (:mod:`gpuwrf.nesting.moving`)
into a REAL driver for the live nested runtime: a ``move`` callback for
:func:`gpuwrf.runtime.domain_tree.run_domain_tree_callbacks` /
``run_operational_domain_tree`` that reproduces WRF's per-move choreography:

1. **decide** at most one parent-cell move per axis -- prescribed moves (the WRF
   preset-moves namelist ``num_moves``/``move_id``/``move_interval``/
   ``move_cd_x``/``move_cd_y``) or vortex-following (``time_for_move2``:
   smoothed 500-hPa height minimum displacement from the nest center);
2. **corral** the move so the nest keeps ``corral_dist`` parent cells clear of
   the parent boundary (WRF ``corral_dist``);
3. **reposition** the edge metadata (``med_nest_move`` updates
   ``i/j_parent_start``);
4. **rebuild** the parent->child forcedown gather weights AND the child->parent
   feedback weights for the new position (WRF re-does its nest communicator /
   interpolation setup after a move; the pre-v0.23 scaffold silently reused the
   stale plans);
5. **shift** the resident child state by ``parent_grid_ratio`` fine cells per
   parent-cell move (``shift_domain_em.F``) and **re-initialize the newly
   exposed rows/columns from the parent** by the same cell-centered SINT-linear
   interpolation the forcedown uses (WRF re-runs the nest-initialization
   interpolation over the exposed region);
6. shift the persistent per-domain scratch carried across steps
   (``*_save``/``ww``/``rthraten``/... in :class:`OperationalCarry`) exactly like
   WRF's registry-wide ``shift_domain_em`` array shift, with parent-interpolated
   fill;
7. leave the boundary package REBUILD to the immediately following forcedown
   (``run_domain_tree_callbacks`` calls ``move`` right before ``force``, which
   re-derives the full specified+relaxation ring from the parent at the NEW
   position through the rebuilt weights -- WRF's ``med_nest_force`` after
   ``med_nest_move``).

Unsupported configurations FAIL CLOSED before compute with a named reason (no
silent fallback): see :class:`MovingNestError` call sites.  The two structural
exclusions this milestone are non-flat child/parent terrain (a move over real
orography requires the WRF terrain/base-state re-blend + ``start_domain``
choreography, not landed) and active prognostic land carries (Noah-MP /
Noah-classic / slab / Pleim-Xiu sub-states would need their own registry-wide
shift).

CPU-lane note: the move decision + weight rebuild are HOST work by design (WRF's
move is host choreography too); the vortex tracker reads three fields back to the
host at ``vortex_interval`` cadence.  Moves are rare O(minutes) events, not
timestep-loop work, so this does not violate the no-in-loop-transfer rule; a GPU
deployment would keep the tracker reduction on-device.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import DomainNest
from gpuwrf.coupling.boundary_feedback import build_state_feedback_weights
from gpuwrf.nesting.boundary_construction import (
    NestForceWeights,
    build_nest_force_weights,
)
from gpuwrf.nesting.moving import (
    _SHIFTABLE_STATE_FIELDS,
    MovingNestBounds,
    NestMove,
    apply_move_to_edge,
    planned_vortex_move,
    shift_array_for_nest_move,
)

G_ACCEL = 9.81


class MovingNestError(ValueError):
    """Fail-closed moving-nest configuration/runtime error with a named reason."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrescribedMove:
    """One WRF preset move: fires right after parent step ``parent_step``.

    Mirrors one entry of the WRF ``&domains`` preset-move arrays
    (``move_interval``/``move_cd_x``/``move_cd_y``), expressed in parent STEPS
    (the runtime clock) instead of minutes.  ``dx/dy`` are parent-cell moves and
    must each be -1, 0, or +1 (WRF moves at most one parent cell per event).
    """

    parent_step: int
    dx_parent: int
    dy_parent: int


@dataclass(frozen=True)
class MovingNestConfig:
    """Opt-in moving-nest configuration for ONE child domain.

    ``mode="prescribed"`` replays ``prescribed_moves``; ``mode="vortex_following"``
    tracks the smoothed ``vortex_level_pa`` geopotential-height minimum in the
    CHILD state every ``move_interval_steps`` parent steps (WRF
    ``time_for_move2`` with ``vortex_interval``).  ``tracker`` optionally
    overrides the built-in tracker with a callable ``child_state -> (i, j)``
    returning the 1-based child-grid vortex center (used by analytic oracles).
    """

    child: str
    mode: str = "prescribed"
    prescribed_moves: tuple[PrescribedMove, ...] = ()
    move_interval_steps: int = 1
    search_radius_child_cells: float = 6.0
    corral_dist: int = 8
    vortex_level_pa: float = 50000.0
    exclude_ring_cells: int = 5
    tracker: Callable[[Any], tuple[float, float]] | None = None
    global_x: bool = False
    global_y: bool = False


_MODES = ("prescribed", "vortex_following")

# Categorical/discrete surface fields: newly exposed cells take the NEAREST
# parent cell's value (WRF interpolates masks with nearest-neighbour, not
# bilinear -- a fractional land mask is meaningless).
_NEAREST_FILL_FIELDS = ("xland", "lakemask")


def validate_moving_nest_config(config: MovingNestConfig) -> None:
    """Fail closed on malformed moving-nest configuration (named reasons)."""

    if config.mode not in _MODES:
        raise MovingNestError(
            f"moving nest {config.child}: unknown mode {config.mode!r}; supported: {_MODES}"
        )
    if int(config.move_interval_steps) < 1:
        raise MovingNestError(
            f"moving nest {config.child}: move_interval_steps must be >= 1"
        )
    if int(config.corral_dist) < 1:
        raise MovingNestError(
            f"moving nest {config.child}: corral_dist must be >= 1 (WRF keeps the nest "
            "clear of the parent specified boundary)"
        )
    if config.mode == "prescribed":
        last = 0
        for entry in config.prescribed_moves:
            if int(entry.parent_step) <= last:
                raise MovingNestError(
                    f"moving nest {config.child}: prescribed_moves must have strictly "
                    f"increasing parent_step (got {entry.parent_step} after {last})"
                )
            last = int(entry.parent_step)
            if abs(int(entry.dx_parent)) > 1 or abs(int(entry.dy_parent)) > 1:
                raise MovingNestError(
                    f"moving nest {config.child}: prescribed move at parent step "
                    f"{entry.parent_step} exceeds one parent cell per axis per event "
                    "(WRF moves at most +-1; split it into multiple events)"
                )
    if config.mode == "vortex_following" and config.tracker is None:
        if float(config.vortex_level_pa) <= 0.0:
            raise MovingNestError(
                f"moving nest {config.child}: vortex_level_pa must be positive"
            )


# ---------------------------------------------------------------------------
# WRF time_for_move2 vortex tracker (host, called at vortex_interval cadence)
# ---------------------------------------------------------------------------


def _smooth121(field: np.ndarray, passes: int = 2) -> np.ndarray:
    """1-2-1 smoother in both horizontal directions (WRF pre-track smoothing)."""

    out = np.asarray(field, dtype=np.float64).copy()
    for _ in range(int(passes)):
        inner = 0.25 * (out[:, :-2] + 2.0 * out[:, 1:-1] + out[:, 2:])
        out[:, 1:-1] = inner
        inner = 0.25 * (out[:-2, :] + 2.0 * out[1:-1, :] + out[2:, :])
        out[1:-1, :] = inner
    return out


def vortex_center_from_state(
    state: Any,
    *,
    level_pa: float = 50000.0,
    exclude_ring_cells: int = 5,
    smooth_passes: int = 2,
) -> tuple[float, float]:
    """Locate the vortex center: minimum smoothed geopotential height at ``level_pa``.

    Follows WRF ``mediation_nest_move.F::time_for_move2``: build the height of
    the target pressure surface (log-p interpolation of the mass-level
    geopotential, from the state's ``p_total``/``ph_total``), smooth it, and take
    the interior minimum (the specified+relaxation ring and one extra cell are
    excluded -- WRF never tracks into the forced frame).  Returns the 1-based
    ``(i, j)`` child-grid center.  Columns are clamped to their top/bottom mass
    level when ``level_pa`` falls outside the column (shallow idealized tops).
    """

    p = np.asarray(state.p_total, dtype=np.float64)
    ph = np.asarray(state.ph_total, dtype=np.float64)
    if p.ndim != 3 or ph.ndim != 3 or ph.shape[0] != p.shape[0] + 1:
        raise MovingNestError(
            "vortex tracker requires 3-D p_total (nz,ny,nx) and ph_total (nz+1,ny,nx); "
            f"got {p.shape} / {ph.shape}"
        )
    z_mass = 0.5 * (ph[:-1] + ph[1:]) / G_ACCEL  # (nz, ny, nx)
    logp = np.log(np.maximum(p, 1.0))
    target = float(np.log(level_pa))
    nz, ny, nx = p.shape
    # Columnwise linear-in-log-p interpolation of z to the target level.  WRF
    # columns have p strictly decreasing with k; np.interp needs increasing x.
    cols = logp.reshape(nz, -1)[::-1]
    zc = z_mass.reshape(nz, -1)[::-1]
    z_level = np.empty(cols.shape[1], dtype=np.float64)
    for idx in range(cols.shape[1]):
        z_level[idx] = np.interp(target, cols[:, idx], zc[:, idx])
    z2d = _smooth121(z_level.reshape(ny, nx), passes=smooth_passes)
    ring = max(1, int(exclude_ring_cells))
    if ny - 2 * ring < 1 or nx - 2 * ring < 1:
        raise MovingNestError(
            f"vortex tracker: child grid {ny}x{nx} too small for exclude_ring_cells={ring}"
        )
    interior = z2d[ring : ny - ring, ring : nx - ring]
    flat = int(np.argmin(interior))
    j = flat // interior.shape[1] + ring
    i = flat % interior.shape[1] + ring
    return float(i + 1), float(j + 1)  # 1-based child-grid center


# ---------------------------------------------------------------------------
# Exposed-region fill: parent -> full child grid at the NEW nest position
# ---------------------------------------------------------------------------


def _nearest_fill_2d(
    parent_field: jax.Array,
    *,
    parent_grid_ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    child_ny: int,
    child_nx: int,
) -> jax.Array:
    """Nearest-neighbour parent->child gather with the SINT cell-centered registration."""

    ratio = int(parent_grid_ratio)

    def axis_idx(start: int, child_len: int, parent_len: int) -> np.ndarray:
        base = float(int(start) - 1)
        idx = np.arange(int(child_len), dtype=np.float64)
        coords = base + (idx - float(ratio // 2)) / float(ratio)
        return np.clip(np.rint(coords), 0, int(parent_len) - 1).astype(np.int32)

    pny, pnx = int(parent_field.shape[-2]), int(parent_field.shape[-1])
    yi = jnp.asarray(axis_idx(j_parent_start, child_ny, pny))
    xi = jnp.asarray(axis_idx(i_parent_start, child_nx, pnx))
    out = jnp.take(jnp.asarray(parent_field), yi, axis=-2)
    return jnp.take(out, xi, axis=-1)


def _stagger_for_trailing_dims(
    shape: tuple[int, ...], child_ny: int, child_nx: int
) -> str | None:
    """Map trailing horizontal dims to C-grid staggering ('mass'/'u'/'v')."""

    if len(shape) < 2:
        return None
    ny, nx = int(shape[-2]), int(shape[-1])
    if ny == child_ny and nx == child_nx:
        return "mass"
    if ny == child_ny and nx == child_nx + 1:
        return "u"
    if ny == child_ny + 1 and nx == child_nx:
        return "v"
    return None


def _interp_by_stagger(field: jax.Array, weights: NestForceWeights, stagger: str) -> jax.Array:
    from gpuwrf.nesting.boundary_construction import interp_parent_field_to_child

    return interp_parent_field_to_child(field, weights, staggering=stagger)


def build_parent_fill_state(
    parent_state: Any,
    child_state: Any,
    new_weights: NestForceWeights,
    *,
    new_spec: DomainNest,
    child_ny: int,
    child_nx: int,
    fields: tuple[str, ...] = _SHIFTABLE_STATE_FIELDS,
) -> SimpleNamespace:
    """Interpolate every shiftable field from the parent to the FULL child grid.

    The result is the ``fill_state`` for
    :func:`gpuwrf.nesting.moving.shift_state_for_nest_move`: newly exposed child
    rows/columns take these parent-interpolated values (WRF re-initializes the
    exposed region from the parent), while the shifted overlap keeps the resident
    child values bit-for-bit.  Staggering is selected from each field's trailing
    dims against the child grid; categorical masks use nearest-neighbour.
    """

    fill: dict[str, jax.Array] = {}
    for name in fields:
        child_value = getattr(child_state, name, None)
        parent_value = getattr(parent_state, name, None)
        if child_value is None or parent_value is None:
            continue
        if getattr(child_value, "ndim", 0) < 2 or getattr(parent_value, "ndim", 0) < 2:
            continue
        stagger = _stagger_for_trailing_dims(tuple(child_value.shape), child_ny, child_nx)
        if stagger is None:
            continue
        if name in _NEAREST_FILL_FIELDS:
            fill[name] = _nearest_fill_2d(
                parent_value,
                parent_grid_ratio=int(new_spec.parent_grid_ratio),
                i_parent_start=int(new_spec.i_parent_start),
                j_parent_start=int(new_spec.j_parent_start),
                child_ny=int(child_value.shape[-2]),
                child_nx=int(child_value.shape[-1]),
            ).astype(child_value.dtype)
        else:
            fill[name] = _interp_by_stagger(parent_value, new_weights, stagger).astype(
                child_value.dtype
            )
    return SimpleNamespace(**fill)


# Persistent OperationalCarry scratch shifted alongside the state (WRF
# shift_domain_em shifts EVERY registry array).  ``*_save`` mirror the state
# leaves; ``ww``/``rthraten``/``t_2ave``/``ph_tend``/``mu*`` persist across steps.
_CARRY_SHIFT_FIELDS = (
    "t_2ave",
    "ww",
    "mudf",
    "muave",
    "muts",
    "ph_tend",
    "u_save",
    "v_save",
    "w_save",
    "t_save",
    "ph_save",
    "mu_save",
    "ww_save",
    "rthraten",
)

# Carry sub-states a registry-wide shift has NOT been implemented for: their
# presence fails closed (silently not shifting a prognostic land column would be
# a wrong-position land state after the move).
_CARRY_FAIL_CLOSED_SUBTREES = (
    "noahmp_land",
    "noahclassic_land",
    "slab_land",
    "px_land",
    "cumulus_carry",
    "base_state",
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class _EdgeTracker:
    """Mutable per-edge move bookkeeping."""

    next_prescribed: int = 0
    moves_applied: int = 0


class MovingNestDriver:
    """Builds the ``move`` callback for the live nested runtime.

    ``geometry`` maps each domain name to an object exposing ``nx``/``ny``
    (a :class:`GridSpec` or any duck-typed stand-in for callback-level gates).
    ``registration`` is the forcedown interpolation registration ("sint" is the
    WRF-faithful default).  Feedback weights are rebuilt on each move whenever
    the pre-move edge carried them.
    """

    def __init__(
        self,
        *,
        geometry: Mapping[str, Any],
        configs: tuple[MovingNestConfig, ...] | list[MovingNestConfig],
        registration: str = "sint",
        feedback_spec_zone: int = 1,
    ) -> None:
        self._geometry = dict(geometry)
        self._registration = str(registration)
        self._feedback_spec_zone = int(feedback_spec_zone)
        self._configs: dict[str, MovingNestConfig] = {}
        for config in configs:
            validate_moving_nest_config(config)
            if config.child in self._configs:
                raise MovingNestError(
                    f"moving nest {config.child}: duplicate MovingNestConfig"
                )
            if config.child not in self._geometry:
                raise MovingNestError(
                    f"moving nest {config.child}: no geometry for child domain"
                )
            self._configs[config.child] = config
        self._edges: dict[tuple[str, str], Any] = {}
        self._trackers: dict[str, _EdgeTracker] = {
            child: _EdgeTracker() for child in self._configs
        }
        self.move_log: list[dict[str, Any]] = []

    # -- construction ------------------------------------------------------

    @classmethod
    def for_tree(
        cls,
        tree: Any,
        configs: tuple[MovingNestConfig, ...] | list[MovingNestConfig],
        *,
        registration: str = "sint",
        feedback_spec_zone: int = 1,
        flat_terrain_tol_m: float = 1.0e-6,
    ) -> "MovingNestDriver":
        """Bind the driver to a live :class:`DomainTree` with fail-closed checks."""

        geometry = {name: bundle.grid for name, bundle in tree.domains.items()}
        driver = cls(
            geometry=geometry,
            configs=configs,
            registration=registration,
            feedback_spec_zone=feedback_spec_zone,
        )
        for config in driver._configs.values():
            parent = tree.hierarchy.parent(config.child)
            if parent is None:
                raise MovingNestError(
                    f"moving nest {config.child}: domain has no parent edge"
                )
            if tree.hierarchy.children(config.child):
                raise MovingNestError(
                    f"moving nest {config.child}: moving a nest that has its own child "
                    "nests is not supported (telescoped moving nests require moving the "
                    "grandchild communicators too; not landed)"
                )
            for name in (parent, config.child):
                grid = tree.domains[name].grid
                terrain = np.asarray(grid.terrain_height)
                if float(np.max(np.abs(terrain))) > float(flat_terrain_tol_m):
                    raise MovingNestError(
                        f"moving nest {config.child}: non-flat terrain on {name} "
                        f"(max |h|={float(np.max(np.abs(terrain))):.3g} m) is not supported; "
                        "a move over orography requires the WRF terrain/base-state re-blend "
                        "+ start_domain re-derivation of the child metrics (not landed)"
                    )
            driver._validate_corral_initial(config, tree_edge_spec(tree, config.child))
        return driver

    def _validate_corral_initial(self, config: MovingNestConfig, spec: DomainNest) -> None:
        lo_i, hi_i, lo_j, hi_j = self._corral_range(spec)
        if not (lo_i <= int(spec.i_parent_start) <= hi_i and lo_j <= int(spec.j_parent_start) <= hi_j):
            raise MovingNestError(
                f"moving nest {config.child}: initial placement i,j=({spec.i_parent_start},"
                f"{spec.j_parent_start}) violates corral_dist={config.corral_dist} "
                f"(allowed i in [{lo_i},{hi_i}], j in [{lo_j},{hi_j}]); shrink corral_dist "
                "or enlarge the parent"
            )

    # -- geometry helpers ----------------------------------------------------

    def _spans(self, spec: DomainNest) -> tuple[int, int]:
        import math

        child = self._geometry[spec.child]
        ratio = int(spec.parent_grid_ratio)
        return (
            max(1, int(math.ceil(int(child.nx) / float(ratio)))),
            max(1, int(math.ceil(int(child.ny) / float(ratio)))),
        )

    def _corral_range(self, spec: DomainNest) -> tuple[int, int, int, int]:
        config = self._configs[spec.child]
        parent = self._geometry[spec.parent]
        span_x, span_y = self._spans(spec)
        corral = int(config.corral_dist)
        lo_i = 1 + corral
        hi_i = int(parent.nx) - span_x - corral + 1
        lo_j = 1 + corral
        hi_j = int(parent.ny) - span_y - corral + 1
        return lo_i, hi_i, lo_j, hi_j

    def _bounds(self, spec: DomainNest) -> MovingNestBounds:
        config = self._configs[spec.child]
        parent = self._geometry[spec.parent]
        child = self._geometry[spec.child]
        return MovingNestBounds(
            parent_nx=int(parent.nx),
            parent_ny=int(parent.ny),
            child_nx=int(child.nx),
            child_ny=int(child.ny),
            parent_grid_ratio=int(spec.parent_grid_ratio),
            global_x=bool(config.global_x),
            global_y=bool(config.global_y),
        )

    # -- move decision -------------------------------------------------------

    def _decide(
        self, config: MovingNestConfig, spec: DomainNest, child_state: Any, parent_step: int
    ) -> NestMove | None:
        tracker = self._trackers[config.child]
        if config.mode == "prescribed":
            moves = config.prescribed_moves
            if tracker.next_prescribed >= len(moves):
                return None
            pending = moves[tracker.next_prescribed]
            if int(pending.parent_step) < int(parent_step):
                raise MovingNestError(
                    f"moving nest {config.child}: prescribed move scheduled for parent step "
                    f"{pending.parent_step} was never dispatched (current parent step "
                    f"{parent_step}); the runtime skipped a move-hook invocation"
                )
            if int(pending.parent_step) != int(parent_step):
                return None
            tracker.next_prescribed += 1
            return NestMove(
                dx_parent=int(pending.dx_parent),
                dy_parent=int(pending.dy_parent),
                reason="prescribed",
            )
        # vortex_following
        if int(parent_step) % int(config.move_interval_steps) != 0:
            return None
        child = self._geometry[config.child]
        if config.tracker is not None:
            vortex_i, vortex_j = config.tracker(child_state)
        else:
            vortex_i, vortex_j = vortex_center_from_state(
                child_state,
                level_pa=float(config.vortex_level_pa),
                exclude_ring_cells=int(config.exclude_ring_cells),
            )
        return planned_vortex_move(
            vortex_i=float(vortex_i),
            vortex_j=float(vortex_j),
            child_nx=int(child.nx),
            child_ny=int(child.ny),
            parent_grid_ratio=int(spec.parent_grid_ratio),
            search_radius_child_cells=float(config.search_radius_child_cells),
        )

    def _corral(self, config: MovingNestConfig, spec: DomainNest, move: NestMove) -> NestMove:
        if bool(config.global_x) and bool(config.global_y):
            return move
        lo_i, hi_i, lo_j, hi_j = self._corral_range(spec)
        dx = int(move.dx_parent)
        dy = int(move.dy_parent)
        if not bool(config.global_x):
            target_i = int(spec.i_parent_start) + dx
            if target_i < lo_i or target_i > hi_i:
                dx = 0
        if not bool(config.global_y):
            target_j = int(spec.j_parent_start) + dy
            if target_j < lo_j or target_j > hi_j:
                dy = 0
        if dx == move.dx_parent and dy == move.dy_parent:
            return move
        return NestMove(dx_parent=dx, dy_parent=dy, reason=f"{move.reason}+corral")

    # -- state/carry movement --------------------------------------------------

    def _move_child_carry(
        self,
        child_carry: Any,
        parent_carry: Any,
        move: NestMove,
        new_spec: DomainNest,
        new_weights: NestForceWeights,
    ) -> Any:
        from gpuwrf.nesting.moving import shift_state_for_nest_move

        config = self._configs[new_spec.child]
        child_state = getattr(child_carry, "state", child_carry)
        parent_state = getattr(parent_carry, "state", parent_carry)
        child = self._geometry[new_spec.child]
        fill_state = build_parent_fill_state(
            parent_state,
            child_state,
            new_weights,
            new_spec=new_spec,
            child_ny=int(child.ny),
            child_nx=int(child.nx),
        )
        moved_state = shift_state_for_nest_move(
            child_state,
            move,
            parent_grid_ratio=int(new_spec.parent_grid_ratio),
            fill_state=fill_state,
            periodic_x=bool(config.global_x),
            periodic_y=bool(config.global_y),
        )
        if child_state is child_carry:
            return moved_state

        for name in _CARRY_FAIL_CLOSED_SUBTREES:
            if getattr(child_carry, name, None) is not None:
                raise MovingNestError(
                    f"moving nest {new_spec.child}: active carry sub-state {name!r} is not "
                    "supported by the moving-nest registry shift (prognostic sub-state "
                    "would be left at the pre-move position); disable it or extend the "
                    "shift registry"
                )

        updates: dict[str, Any] = {"state": moved_state}
        for name in _CARRY_SHIFT_FIELDS:
            child_value = getattr(child_carry, name, None)
            if child_value is None or getattr(child_value, "ndim", 0) < 2:
                continue
            stagger = _stagger_for_trailing_dims(
                tuple(child_value.shape), int(child.ny), int(child.nx)
            )
            if stagger is None:
                raise MovingNestError(
                    f"moving nest {new_spec.child}: carry scratch {name!r} with shape "
                    f"{tuple(child_value.shape)} does not match any child staggering; "
                    "refusing to leave it unshifted"
                )
            parent_value = getattr(parent_carry, name, None)
            fill = (
                _interp_by_stagger(parent_value, new_weights, stagger).astype(child_value.dtype)
                if parent_value is not None and getattr(parent_value, "ndim", 0) >= 2
                else None
            )
            updates[name] = shift_array_for_nest_move(
                child_value,
                move,
                parent_grid_ratio=int(new_spec.parent_grid_ratio),
                fill=fill,
                periodic_x=bool(config.global_x),
                periodic_y=bool(config.global_y),
            )
        return child_carry.replace(**updates)

    # -- the runtime hook -------------------------------------------------------

    def make_move_fn(self):
        """Return the ``move`` callback for ``run_domain_tree_callbacks``."""

        from gpuwrf.runtime.domain_tree import DomainEdge

        def move(edge: Any, parent_carry: Any, child_carry: Any, parent_step: int):
            config = self._configs.get(edge.spec.child)
            if config is None:
                return None  # static child: untouched default path
            key = (edge.spec.parent, edge.spec.child)
            current = self._edges.get(key, edge)
            child_state = getattr(child_carry, "state", child_carry)
            planned = self._decide(config, current.spec, child_state, int(parent_step))
            if planned is None or not planned.moved:
                # No move this step: keep the (possibly previously moved) edge live.
                return current if current is not edge else None
            corralled = self._corral(config, current.spec, planned)
            if not corralled.moved:
                self.move_log.append(
                    {
                        "child": edge.spec.child,
                        "parent_step": int(parent_step),
                        "planned": (planned.dx_parent, planned.dy_parent),
                        "applied": (0, 0),
                        "reason": corralled.reason,
                        "outcome": "corral_blocked",
                    }
                )
                return current if current is not edge else None
            new_spec = apply_move_to_edge(
                current.spec, corralled, bounds=self._bounds(current.spec)
            )
            if (
                new_spec.i_parent_start == current.spec.i_parent_start
                and new_spec.j_parent_start == current.spec.j_parent_start
            ):
                return current if current is not edge else None
            parent_grid = self._geometry[new_spec.parent]
            child_grid = self._geometry[new_spec.child]
            new_weights = build_nest_force_weights(
                parent_grid_ratio=int(new_spec.parent_grid_ratio),
                i_parent_start=int(new_spec.i_parent_start),
                j_parent_start=int(new_spec.j_parent_start),
                parent_grid=parent_grid,
                child_grid=child_grid,
                registration=self._registration,
            )
            new_feedback = (
                build_state_feedback_weights(
                    parent_grid_ratio=int(new_spec.parent_grid_ratio),
                    i_parent_start=int(new_spec.i_parent_start),
                    j_parent_start=int(new_spec.j_parent_start),
                    parent_grid=parent_grid,
                    child_grid=child_grid,
                    spec_zone=self._feedback_spec_zone,
                )
                if current.feedback_weights is not None
                else None
            )
            moved_carry = self._move_child_carry(
                child_carry, parent_carry, corralled, new_spec, new_weights
            )
            new_edge = DomainEdge(
                spec=new_spec, weights=new_weights, feedback_weights=new_feedback
            )
            self._edges[key] = new_edge
            self._trackers[edge.spec.child].moves_applied += 1
            self.move_log.append(
                {
                    "child": edge.spec.child,
                    "parent_step": int(parent_step),
                    "planned": (planned.dx_parent, planned.dy_parent),
                    "applied": (corralled.dx_parent, corralled.dy_parent),
                    "from": (current.spec.i_parent_start, current.spec.j_parent_start),
                    "to": (new_spec.i_parent_start, new_spec.j_parent_start),
                    "reason": corralled.reason,
                    "outcome": "moved",
                }
            )
            return new_edge, moved_carry

        return move


def tree_edge_spec(tree: Any, child: str) -> DomainNest:
    """Static tree edge spec for ``child`` (helper for validation and tests)."""

    for edge in tree.hierarchy.nests:
        if edge.child == child:
            return edge
    raise MovingNestError(f"no nest edge found for child domain {child!r}")


__all__ = [
    "MovingNestConfig",
    "MovingNestDriver",
    "MovingNestError",
    "PrescribedMove",
    "build_parent_fill_state",
    "tree_edge_spec",
    "validate_moving_nest_config",
    "vortex_center_from_state",
]
