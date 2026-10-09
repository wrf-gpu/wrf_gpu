"""WRF nest radiation ozone: a child's o3rad is the parent's held o3rad, SINT-interpolated level-wise.

WRF (o3input=2) interpolates the CAM ozone climatology only on domain 1 (module_radiation_driver.F:1803,
``IF (o3input .EQ. 2 .AND. id .EQ. 1)``).  Nests receive ``o3rad`` from their parent through Registry
``rdf=(p2c)`` (Registry.EM_COMMON:1264): the force-down after EVERY parent step calls p2c -> interp_fcn
(inc/nest_forcedown_interp.inc:265-280, imask_nostag = 1, interp_method_type SINT, mass point), with no
re-interpolation to the nest pressure; d03 receives d02's copy.  ``GPUWRF_NEST_O3_FROM_PARENT=1`` (default
off) selects this behaviour; the flag is read on the host when the domain carries are built.
"""

from __future__ import annotations

import os

import jax

from gpuwrf.nesting.interp import interp_sint_full

NEST_O3_FROM_PARENT_ENV = "GPUWRF_NEST_O3_FROM_PARENT"


def nest_o3_from_parent_enabled() -> bool:
    """Host-side switch, resolved when the nested carries are built (never inside traced code)."""

    return os.environ.get(NEST_O3_FROM_PARENT_ENV, "0") == "1"


def force_child_o3rad(parent_o3rad, mass_weights, *, parent_grid_ratio: int):
    """WRF force-down of ``o3rad``: SINT of the parent field onto the full child mass grid, level by level."""

    out = interp_sint_full(parent_o3rad, mass_weights, parent_grid_ratio=int(parent_grid_ratio))
    return out.astype(parent_o3rad.dtype)


_COMPILED: dict = {}


def _compiled_force(mass_weights, parent_grid_ratio: int, batched: bool):
    """One jitted force per edge (weights closed over, like the per-producer force-down jits)."""

    key = (id(mass_weights), int(parent_grid_ratio), bool(batched))
    hit = _COMPILED.get(key)
    if hit is None or hit[0] is not mass_weights:
        def one(field):
            return force_child_o3rad(field, mass_weights, parent_grid_ratio=parent_grid_ratio)

        hit = (mass_weights, jax.jit(jax.vmap(one) if batched else one))
        _COMPILED[key] = hit
    return hit[1]


def force_child_carry_o3rad(child_carry, parent_carry, weights, *, parent_grid_ratio: int, batched: bool = False):
    """Return ``child_carry`` with its held ``o3rad`` forced from ``parent_carry``; unchanged when either has none.

    ``batched`` maps over a leading case axis (the batched domain tree carries ``(B, z, y, x)``).
    """

    child_o3 = getattr(child_carry, "o3rad", None)
    parent_o3 = getattr(parent_carry, "o3rad", None)
    if child_o3 is None or parent_o3 is None:
        return child_carry
    forced = _compiled_force(weights.mass, int(parent_grid_ratio), bool(batched))(parent_o3)
    return child_carry.replace(o3rad=forced.astype(child_o3.dtype))


__all__ = [
    "NEST_O3_FROM_PARENT_ENV",
    "force_child_carry_o3rad",
    "force_child_o3rad",
    "nest_o3_from_parent_enabled",
]
