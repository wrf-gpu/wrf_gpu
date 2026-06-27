"""WRF-faithful live-nesting construction for the GPU port (P0-1a).

This package implements, against recorded/controlled parent states, the three
WRF down-nesting pieces:

  * :mod:`gpuwrf.nesting.interp` -- parent->child spatial interpolation operators
    (``share/interp_fcn.F`` / ``share/sint.F``): the WRF cell-centered ``sint``
    registration (default) and the node-aligned bilinear baseline, plus a host
    monotone-TR4 ``sint`` reference for fidelity measurement.
  * :mod:`gpuwrf.nesting.boundary_construction` -- the child specified+relaxation
    ``*_bdy`` package built from a parent state (WRF ``med_nest_force`` /
    ``bdy_interp1``), matching the ``State.*_bdy`` / ``boundary_apply`` interface.
  * :mod:`gpuwrf.nesting.scheduler` -- the parent->child subcycling cadence and
    the ``med_nest_force`` forcedown ordering (WRF ``frame/module_integrate.F``),
    pure host; the runtime device hook is SPEC'd for the manager (P0-1b).

P0-1a SCOPE: prove construction + interpolation + scheduler are WRF-faithful vs
recorded parent states.  NO live-skill claim, NO in-loop host/device transfer
(both are P0-1b, after P0-6 + P0-4 close).
"""

from gpuwrf.nesting.interp import (
    InterpWeights,
    build_bilinear_weights,
    build_sint_weights,
    interp_bilinear,
    interp_sint_linear,
    sint_block_reference,
    sint_to_child_reference,
)
from gpuwrf.nesting.boundary_construction import (
    NestForceWeights,
    build_nest_force_weights,
    build_child_boundary_package,
    interp_parent_field_to_child,
    field_sides_3d,
    field_sides_2d,
    field_sides_3d_edgeonly,
    field_sides_2d_edgeonly,
)
from gpuwrf.nesting.scheduler import (
    NestEdge,
    NestTower,
    expected_substep_counts,
    forcedown_event_log,
    run_host_tower,
    runtime_hook_spec,
)
from gpuwrf.nesting.moving import (
    MovingNestBounds,
    NestMove,
    apply_move_to_edge,
    planned_vortex_move,
    shift_array_for_nest_move,
    shift_state_for_nest_move,
)
from gpuwrf.nesting.adaptive_timestep import (
    AdaptiveTimeStepConfig,
    AdaptiveTimeStepResult,
    AdaptiveTimeStepState,
    adapt_timestep,
    calc_dt_candidate,
)

__all__ = [
    "InterpWeights",
    "build_bilinear_weights",
    "build_sint_weights",
    "interp_bilinear",
    "interp_sint_linear",
    "sint_block_reference",
    "sint_to_child_reference",
    "NestForceWeights",
    "build_nest_force_weights",
    "build_child_boundary_package",
    "interp_parent_field_to_child",
    "field_sides_3d",
    "field_sides_2d",
    "field_sides_3d_edgeonly",
    "field_sides_2d_edgeonly",
    "NestEdge",
    "NestTower",
    "expected_substep_counts",
    "forcedown_event_log",
    "run_host_tower",
    "runtime_hook_spec",
    "MovingNestBounds",
    "NestMove",
    "apply_move_to_edge",
    "planned_vortex_move",
    "shift_array_for_nest_move",
    "shift_state_for_nest_move",
    "AdaptiveTimeStepConfig",
    "AdaptiveTimeStepResult",
    "AdaptiveTimeStepState",
    "adapt_timestep",
    "calc_dt_candidate",
]
