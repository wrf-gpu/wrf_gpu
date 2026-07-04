"""Operational adaptive-timestep driver (WRF ``adapt_timestep_em.F``, v0.23 G2).

Binds the v0.22 :mod:`gpuwrf.nesting.adaptive_timestep` planner (a faithful
``calc_dt`` port) to the LIVE operational domain tree:

* CFL diagnostics are computed from the ROOT domain's resident state after each
  step (WRF tracks ``max_vert_cfl``/``max_horiz_cfl`` inside the RK loop; this
  driver reads the equivalent end-of-step diagnostic -- same quantities, one
  step of latency, documented deviation);
* the planner proposes the next root ``dt`` (growth-capped, rounded to the WRF
  ``precision`` quantum, min/max clamped);
* the new ``dt`` is written through to the root namelist and CASCADED down the
  tree: every child keeps WRF's fixed nest ratio (``child_dt = parent_dt /
  parent_grid_ratio`` -- the runtime's subcycle count is structurally
  ``parent_grid_ratio``) and each child's two-time boundary interpolation
  cadence is re-pinned to the new parent ``dt`` (the WRF ``dtbc`` cadence; a
  stale cadence would mis-time the boundary relaxation after a dt change).

Compile-variant honesty: ``dt_s`` is a static cache key of the compiled advance,
so each DISTINCT quantized dt value costs one cold compile per domain (the WRF
``precision`` rounding bounds the variant count; pass a coarse quantum, e.g.
``precision=4`` = quarter seconds, to bound it further).  This is the CPU-lane
implementation; folding dt into a traced scalar to avoid recompiles is the
tracked GPU follow-up.

Host-transfer honesty: the CFL reduction reads ``u/v/w/ph_total`` back to the
host once per ROOT step.  That is acceptable on the CPU lane only; a GPU
deployment must fold the reduction on-device (ADR no-in-loop-transfer rule).
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import Any

import numpy as np

from gpuwrf.nesting.adaptive_timestep import (
    AdaptiveTimeStepConfig,
    AdaptiveTimeStepState,
    adapt_timestep,
)

G_ACCEL = 9.81


def cfl_diagnostics_from_state(
    state: Any, *, dt_s: float, dx_m: float, dy_m: float
) -> tuple[float, float]:
    """Return ``(max_horiz_cfl, max_vert_cfl)`` for ``state`` at ``dt_s``.

    Horizontal: ``max over mass cells of dt*(|u_c|/dx + |v_c|/dy)`` with C-grid
    face winds averaged to cell centers.  Vertical: ``max over layers of
    dt*|w_c|/dz`` with the layer thickness from the resident total geopotential.
    Host NumPy reductions (see module docstring for the transfer policy).
    """

    dt = float(dt_s)
    u = np.asarray(state.u, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    w = np.asarray(state.w, dtype=np.float64)
    ph = np.asarray(state.ph_total, dtype=np.float64)
    u_c = 0.5 * (u[..., :-1] + u[..., 1:])
    v_c = 0.5 * (v[..., :-1, :] + v[..., 1:, :])
    horiz = float(np.max(np.abs(u_c) / float(dx_m) + np.abs(v_c) / float(dy_m))) * dt
    z_face = ph / G_ACCEL
    dz = np.maximum(z_face[1:] - z_face[:-1], 1.0e-3)  # (nz, ny, nx) layer depth
    w_c = 0.5 * (w[:-1] + w[1:])
    vert = float(np.max(np.abs(w_c) / dz)) * dt
    return horiz, vert


class AdaptiveDtDriver:
    """Builds the ``adaptive_dt`` callback for ``run_operational_domain_tree``.

    The ROOT domain adapts; children stay on the WRF fixed-ratio divisor.  The
    driver rewrites ``tree.domains`` namelists in place (the operational advance
    reads the CURRENT bundle namelist on every call, so the next chunk picks the
    new dt up without touching the compiled-program plumbing).
    """

    def __init__(self, tree: Any, config: AdaptiveTimeStepConfig) -> None:
        self._tree = tree
        self._config = config
        self._root = tree.root()
        root_nl = tree.domains[self._root].namelist
        self._last_dt = float(root_nl.dt_s)
        self._current_seconds = 0.0
        self.dt_log: list[dict[str, Any]] = []
        # Pin every child's dt/cadence to the CURRENT root dt up front so a
        # mis-built tree (child dt != parent dt / ratio) is corrected before
        # the first step rather than silently integrated.
        self._cascade(self._root, float(root_nl.dt_s))

    # -- namelist write-through ------------------------------------------------

    def _set_domain_dt(self, name: str, dt_s: float) -> None:
        bundle = self._tree.domains[name]
        if abs(float(bundle.namelist.dt_s) - float(dt_s)) > 0.0:
            namelist = dataclass_replace(bundle.namelist, dt_s=float(dt_s))
            self._tree.domains[name] = dataclass_replace(bundle, namelist=namelist)

    def _cascade(self, parent_name: str, parent_dt_s: float) -> None:
        from gpuwrf.runtime.domain_tree import with_live_child_boundary_config

        self._set_domain_dt(parent_name, float(parent_dt_s))
        for edge in self._tree.children(parent_name):
            child_dt = float(parent_dt_s) / float(edge.parent_grid_ratio)
            bundle = self._tree.domains[edge.child]
            namelist = dataclass_replace(bundle.namelist, dt_s=child_dt)
            namelist = with_live_child_boundary_config(
                namelist,
                parent_dt_s=float(parent_dt_s),
                nested_ph_relax=bool(namelist.boundary_config.nested_ph_relax),
                nested_w_relax=bool(namelist.boundary_config.nested_w_relax),
                nested_ph_spec=bool(namelist.boundary_config.nested_ph_spec),
            )
            self._tree.domains[edge.child] = dataclass_replace(bundle, namelist=namelist)
            self._cascade(edge.child, child_dt)

    # -- the runtime hook --------------------------------------------------------

    def make_adaptive_dt_fn(self):
        def adaptive_dt(name: str, carry: Any, start_step: int):
            if name != self._root:
                return None  # children are pinned by the cascade
            namelist = self._tree.domains[name].namelist
            current_dt = float(namelist.dt_s)
            state = getattr(carry, "state", carry)
            grid = self._tree.domains[name].grid
            horiz, vert = cfl_diagnostics_from_state(
                state,
                dt_s=current_dt,
                dx_m=float(grid.projection.dx_m),
                dy_m=float(grid.projection.dy_m),
            )
            result = adapt_timestep(
                AdaptiveTimeStepState(
                    dt_s=current_dt,
                    last_dt_s=self._last_dt,
                    max_vert_cfl=vert,
                    max_horiz_cfl=horiz,
                    current_seconds=self._current_seconds,
                    advance_count=int(start_step) - 1,
                ),
                self._config,
            )
            if abs(float(result.dt_s) - current_dt) > 0.0:
                self._cascade(name, float(result.dt_s))
            self._last_dt = float(result.last_dt_s)
            self._current_seconds += float(result.dt_s)
            self.dt_log.append(
                {
                    "step": int(start_step),
                    "dt_s": float(result.dt_s),
                    "max_horiz_cfl": horiz,
                    "max_vert_cfl": vert,
                    "reason": result.reason,
                }
            )
            return carry, float(result.dt_s)

        return adaptive_dt


__all__ = ["AdaptiveDtDriver", "cfl_diagnostics_from_state"]
