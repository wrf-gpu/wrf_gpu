"""WRF-referenced moving-nest metadata and resident state-shift helpers.

This is the small, reusable core of WRF's moving-nest path:

* ``mediation_nest_move.F::time_for_move2`` chooses at most one parent-cell move
  in each horizontal direction from a vortex displacement;
* ``med_nest_move`` updates ``i_parent_start`` / ``j_parent_start`` by that
  parent-cell displacement;
* ``shift_domain_em.F`` shifts the child state by ``parent_grid_ratio`` fine-grid
  cells per parent-cell move, then WRF reinitializes newly exposed cells from the
  parent interpolation path.

The helpers here keep that split explicit.  They do not try to reproduce WRF's
full terrain reblend / restart / MPI communicator choreography in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import DomainNest


@dataclass(frozen=True)
class NestMove:
    """A WRF moving-nest displacement in parent-domain cells."""

    dx_parent: int = 0
    dy_parent: int = 0
    reason: str = "manual"

    def __post_init__(self) -> None:
        dx = max(-1, min(1, int(self.dx_parent)))
        dy = max(-1, min(1, int(self.dy_parent)))
        object.__setattr__(self, "dx_parent", dx)
        object.__setattr__(self, "dy_parent", dy)
        object.__setattr__(self, "reason", str(self.reason))

    @property
    def moved(self) -> bool:
        return bool(self.dx_parent or self.dy_parent)


@dataclass(frozen=True)
class MovingNestBounds:
    """Parent/child bounds used to keep or wrap a moved nest start."""

    parent_nx: int | None = None
    parent_ny: int | None = None
    child_nx: int | None = None
    child_ny: int | None = None
    parent_grid_ratio: int = 1
    global_x: bool = False
    global_y: bool = False

    def __post_init__(self) -> None:
        if int(self.parent_grid_ratio) <= 0:
            raise ValueError("parent_grid_ratio must be positive")
        object.__setattr__(self, "parent_grid_ratio", int(self.parent_grid_ratio))


_SHIFTABLE_STATE_FIELDS: tuple[str, ...] = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "p_total",
    "p_perturbation",
    "ph_total",
    "ph_perturbation",
    "mu_total",
    "mu_perturbation",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "Ni",
    "Nr",
    "Ns",
    "Ng",
    "qke",
    "ustar",
    "theta_flux",
    "qv_flux",
    "tau_u",
    "tau_v",
    "rhosfc",
    "fltv",
    "t_skin",
    "soil_moisture",
    "xland",
    "lakemask",
    "mavail",
    "roughness_m",
    "rain_acc",
    "snow_acc",
    "graupel_acc",
    "ice_acc",
    "Nc",
    "Nn",
    "rainc_acc",
    "qsq",
    "qc_bl",
    "qi_bl",
    "cldfra_bl",
    "qh",
    "Nh",
    "qvolg",
    "qvolh",
    "nwfa",
    "nifa",
    "hail_acc",
)


def planned_vortex_move(
    *,
    vortex_i: float,
    vortex_j: float,
    child_nx: int,
    child_ny: int,
    parent_grid_ratio: int,
    search_radius_child_cells: float = 6.0,
) -> NestMove:
    """Return WRF's one-parent-cell vortex-following move decision.

    Mirrors the decision core in ``time_for_move2`` after the vortex center has
    been found: displacement from the child-domain center is clipped to the WRF
    six-cell search radius, divided by ``parent_grid_ratio``, truncated toward
    zero, then clipped to ``[-1, 1]``.
    """

    ratio = int(parent_grid_ratio)
    if ratio <= 0:
        raise ValueError("parent_grid_ratio must be positive")
    center_i = (int(child_nx) + 1) / 2.0
    center_j = (int(child_ny) + 1) / 2.0
    disp_x = max(-float(search_radius_child_cells), min(float(search_radius_child_cells), float(vortex_i) - center_i))
    disp_y = max(-float(search_radius_child_cells), min(float(search_radius_child_cells), float(vortex_j) - center_j))
    return NestMove(
        dx_parent=max(-1, min(1, int(disp_x / float(ratio)))),
        dy_parent=max(-1, min(1, int(disp_y / float(ratio)))),
        reason="vortex_following",
    )


def _child_parent_span(child_len: int | None, ratio: int) -> int:
    if child_len is None:
        return 1
    return max(1, int(math.ceil(float(child_len) / float(ratio))))


def _bounded_start(
    start_1based: int,
    *,
    parent_len: int | None,
    child_len: int | None,
    ratio: int,
    global_wrap: bool,
) -> int:
    if parent_len is None or child_len is None:
        return max(1, int(start_1based))
    max_start = max(1, int(parent_len) - _child_parent_span(child_len, ratio) + 1)
    start = int(start_1based)
    if bool(global_wrap):
        return ((start - 1) % max_start) + 1
    return max(1, min(max_start, start))


def apply_move_to_edge(
    edge: DomainNest,
    move: NestMove,
    *,
    bounds: MovingNestBounds | None = None,
) -> DomainNest:
    """Return ``edge`` with its WRF parent-start metadata moved.

    WRF updates ``i_parent_start`` / ``j_parent_start`` by the parent-cell move.
    For regional nests, starts are clamped to the valid parent footprint.  For a
    global/periodic parent, the matching axis wraps instead; this is the explicit
    scaffold for global nests and avoids accidental non-periodic clipping.
    """

    ratio = int(edge.parent_grid_ratio)
    b = bounds or MovingNestBounds(parent_grid_ratio=ratio)
    new_i = _bounded_start(
        int(edge.i_parent_start) + int(move.dx_parent),
        parent_len=b.parent_nx,
        child_len=b.child_nx,
        ratio=ratio,
        global_wrap=bool(b.global_x),
    )
    new_j = _bounded_start(
        int(edge.j_parent_start) + int(move.dy_parent),
        parent_len=b.parent_ny,
        child_len=b.child_ny,
        ratio=ratio,
        global_wrap=bool(b.global_y),
    )
    return DomainNest(
        edge.parent,
        edge.child,
        ratio,
        new_i,
        new_j,
        feedback=bool(edge.feedback),
    )


def _fill_exposed_regions(
    shifted: jax.Array,
    fill: jax.Array,
    *,
    x_shift: int,
    y_shift: int,
    periodic_x: bool,
    periodic_y: bool,
) -> jax.Array:
    out = shifted
    nx = int(out.shape[-1])
    ny = int(out.shape[-2])
    if x_shift > 0 and not periodic_x:
        width = min(x_shift, nx)
        out = out.at[..., nx - width : nx].set(fill[..., nx - width : nx])
    elif x_shift < 0 and not periodic_x:
        width = min(-x_shift, nx)
        out = out.at[..., :width].set(fill[..., :width])
    if y_shift > 0 and not periodic_y:
        width = min(y_shift, ny)
        out = out.at[..., ny - width : ny, :].set(fill[..., ny - width : ny, :])
    elif y_shift < 0 and not periodic_y:
        width = min(-y_shift, ny)
        out = out.at[..., :width, :].set(fill[..., :width, :])
    return out


def shift_array_for_nest_move(
    field,
    move: NestMove,
    *,
    parent_grid_ratio: int,
    fill=None,
    fill_value: float = 0.0,
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> jax.Array:
    """Shift one field by the fine-grid displacement implied by ``move``.

    A positive parent move shifts overlapping old values toward lower child
    indices by ``parent_grid_ratio`` fine cells; newly exposed high-index cells
    are filled from ``fill`` when provided, otherwise from ``fill_value``.  With a
    global/periodic axis the shift wraps on that axis.
    """

    arr = jnp.asarray(field)
    if arr.ndim < 2:
        return arr
    ratio = int(parent_grid_ratio)
    if ratio <= 0:
        raise ValueError("parent_grid_ratio must be positive")
    x_shift = int(move.dx_parent) * ratio
    y_shift = int(move.dy_parent) * ratio
    if x_shift == 0 and y_shift == 0:
        return arr
    fill_arr = jnp.asarray(fill, dtype=arr.dtype) if fill is not None else jnp.full_like(arr, fill_value)
    shifted = jnp.roll(arr, shift=(-y_shift, -x_shift), axis=(-2, -1))
    return _fill_exposed_regions(
        shifted,
        fill_arr,
        x_shift=x_shift,
        y_shift=y_shift,
        periodic_x=bool(periodic_x),
        periodic_y=bool(periodic_y),
    ).astype(arr.dtype)


def shift_state_for_nest_move(
    state,
    move: NestMove,
    *,
    parent_grid_ratio: int,
    fill_state=None,
    fill_value: float = 0.0,
    fields: Iterable[str] = _SHIFTABLE_STATE_FIELDS,
    periodic_x: bool = False,
    periodic_y: bool = False,
):
    """Shift the resident child state after a moving-nest metadata update.

    Boundary packages are deliberately excluded; WRF rebuilds live child
    boundaries from the parent after a move.  ``fill_state`` may be a freshly
    parent-interpolated child state and supplies the newly exposed rows/columns.
    """

    updates: dict[str, jax.Array] = {}
    for name in fields:
        if not hasattr(state, name):
            continue
        value = getattr(state, name)
        if value is None or getattr(value, "ndim", 0) < 2:
            continue
        fill = getattr(fill_state, name) if fill_state is not None and hasattr(fill_state, name) else None
        updates[name] = shift_array_for_nest_move(
            value,
            move,
            parent_grid_ratio=int(parent_grid_ratio),
            fill=fill,
            fill_value=float(fill_value),
            periodic_x=periodic_x,
            periodic_y=periodic_y,
        )
    if not updates:
        return state
    try:
        return state.replace(_cast=False, **updates)
    except TypeError:
        return state.replace(**updates)


__all__ = [
    "MovingNestBounds",
    "NestMove",
    "apply_move_to_edge",
    "planned_vortex_move",
    "shift_array_for_nest_move",
    "shift_state_for_nest_move",
]
