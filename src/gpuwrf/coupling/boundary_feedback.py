"""Child->parent two-way nest feedback operators.

This module ports the v0.5.0 WRF ``copy_fcn`` feedback operator onto the current
v0.10.0 state contract.  It is intentionally separate from the default
one-way live nesting path: callers must opt in with an explicit feedback gate.

The implemented geometric core is the WRF odd-ratio area average:

* mass fields: each parent cell receives the mean of the ``ratio**2`` child mass
  cells that tile it;
* U faces: each parent U face receives the mean of the coincident child U faces
  along ``j``;
* V faces: each parent V face receives the mean of the coincident child V faces
  along ``i``.

The child specified boundary ring is excluded from the feedback overlap.  The
per-call device op is a static ``jnp.take`` plus mean plus scatter; the gather
indices are precomputed once from the static domain geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State


@dataclass(frozen=True)
class FeedbackWeights:
    """Static WRF ``copy_fcn`` gather plan for one horizontal staggering."""

    parent_lin: jax.Array
    child_lin: jax.Array
    pwe: int
    psn: int
    cwe: int
    csn: int
    stencil: int
    ratio: int
    stagger: str

    def tree_flatten(self):
        return (self.parent_lin, self.child_lin), (
            self.pwe,
            self.psn,
            self.cwe,
            self.csn,
            self.stencil,
            self.ratio,
            self.stagger,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        parent_lin, child_lin = children
        pwe, psn, cwe, csn, stencil, ratio, stagger = aux
        return cls(parent_lin, child_lin, pwe, psn, cwe, csn, stencil, ratio, stagger)


jax.tree_util.register_pytree_node_class(FeedbackWeights)


@dataclass(frozen=True)
class StateFeedbackWeights:
    """Feedback plans for mass, U-face, and V-face leaves of one nest edge."""

    mass: FeedbackWeights
    u: FeedbackWeights
    v: FeedbackWeights

    def tree_flatten(self):
        return (self.mass, self.u, self.v), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        mass, u, v = children
        return cls(mass=mass, u=u, v=v)


jax.tree_util.register_pytree_node_class(StateFeedbackWeights)


@dataclass(frozen=True)
class ConservationResult:
    """Integral conservation diagnostic for one feedback field."""

    leaf: str
    stagger: str
    n_cells: int
    child_overlap_integral: float
    parent_overlap_integral: float
    abs_residual: float
    rel_residual: float
    conserved: bool


def _parent_child_extents(
    *,
    parent_we: int,
    parent_sn: int,
    child_we: int,
    child_sn: int,
    stagger: str,
) -> tuple[int, int, int, int]:
    pwe = int(parent_we) + (1 if stagger == "U" else 0)
    psn = int(parent_sn) + (1 if stagger == "V" else 0)
    cwe = int(child_we) + (1 if stagger == "U" else 0)
    csn = int(child_sn) + (1 if stagger == "V" else 0)
    return pwe, psn, cwe, csn


def build_feedback_weights(
    *,
    parent_grid_ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    parent_we: int,
    parent_sn: int,
    child_we: int,
    child_sn: int,
    stagger: str = "",
    spec_zone: int = 1,
) -> FeedbackWeights:
    """Precompute WRF ``copy_fcn`` child->parent feedback weights.

    ``parent_we``/``parent_sn`` and ``child_we``/``child_sn`` are mass-grid
    horizontal extents.  ``stagger`` is ``""`` for mass/w fields, ``"U"`` for
    U faces, and ``"V"`` for V faces.
    """

    if stagger not in ("", "U", "V"):
        raise ValueError(f"unknown feedback stagger {stagger!r}")
    ratio = int(parent_grid_ratio)
    if ratio <= 1:
        raise ValueError(f"feedback requires parent_grid_ratio > 1, got {ratio}")
    if ratio % 2 == 0:
        raise NotImplementedError("feedback currently supports the WRF odd-ratio copy_fcn branch")

    ipos = int(i_parent_start)
    jpos = int(j_parent_start)
    sz = int(spec_zone)
    pwe, psn, cwe, csn = _parent_child_extents(
        parent_we=parent_we,
        parent_sn=parent_sn,
        child_we=child_we,
        child_sn=child_sn,
        stagger=stagger,
    )

    istag = 0 if stagger == "U" else 1
    jstag = 0 if stagger == "V" else 1
    n_parent_i = int(child_we) // ratio
    n_parent_j = int(child_sn) // ratio

    cj_lo = jpos + sz
    cj_hi = jpos + n_parent_j - jstag - sz
    ci_lo = ipos + sz
    ci_hi = ipos + n_parent_i - istag - sz
    stencil = ratio * ratio if stagger == "" else ratio

    parent_lin: list[int] = []
    child_lin: list[list[int]] = []
    for cj in range(cj_lo, cj_hi + 1):
        for ci in range(ci_lo, ci_hi + 1):
            if stagger == "":
                ni = (ci - ipos) * ratio + ratio // 2 + 1
                nj = (cj - jpos) * ratio + ratio // 2 + 1
                donors = []
                for jj in range(ratio):
                    jp = jj - ratio // 2
                    for ii in range(ratio):
                        ip = ii - ratio // 2
                        donors.append((nj + jp - 1) * cwe + (ni + ip - 1))
            elif stagger == "U":
                ni = (ci - ipos) * ratio + 1
                nj = (cj - jpos) * ratio + ratio // 2 + 1
                donors = []
                for jj in range(ratio):
                    jp = jj - ratio // 2
                    donors.append((nj + jp - 1) * cwe + (ni - 1))
            else:
                ni = (ci - ipos) * ratio + ratio // 2 + 1
                nj = (cj - jpos) * ratio + 1
                donors = []
                for ii in range(ratio):
                    ip = ii - ratio // 2
                    donors.append((nj - 1) * cwe + (ni + ip - 1))

            for donor in donors:
                drow, dcol = divmod(donor, cwe)
                if not (0 <= drow < csn and 0 <= dcol < cwe):
                    raise IndexError(
                        f"feedback donor out of bounds: child=({drow},{dcol}) extent=({csn},{cwe})"
                    )
            parent_lin.append((cj - 1) * pwe + (ci - 1))
            child_lin.append(donors)

    child_arr = np.asarray(child_lin, dtype=np.int32).reshape((-1, stencil))
    return FeedbackWeights(
        parent_lin=jnp.asarray(np.asarray(parent_lin, dtype=np.int32)),
        child_lin=jnp.asarray(child_arr),
        pwe=pwe,
        psn=psn,
        cwe=cwe,
        csn=csn,
        stencil=stencil,
        ratio=ratio,
        stagger=stagger,
    )


def build_state_feedback_weights(
    *,
    parent_grid_ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    parent_grid: GridSpec,
    child_grid: GridSpec,
    spec_zone: int = 1,
) -> StateFeedbackWeights:
    """Build mass/U/V feedback weights for a full ``State`` edge."""

    common = dict(
        parent_grid_ratio=int(parent_grid_ratio),
        i_parent_start=int(i_parent_start),
        j_parent_start=int(j_parent_start),
        parent_we=int(parent_grid.nx),
        parent_sn=int(parent_grid.ny),
        child_we=int(child_grid.nx),
        child_sn=int(child_grid.ny),
        spec_zone=int(spec_zone),
    )
    return StateFeedbackWeights(
        mass=build_feedback_weights(stagger="", **common),
        u=build_feedback_weights(stagger="U", **common),
        v=build_feedback_weights(stagger="V", **common),
    )


def _feedback_values(child: jax.Array, weights: FeedbackWeights) -> jax.Array:
    child_idx = weights.child_lin.reshape(-1)
    if child.ndim == 3:
        z = int(child.shape[0])
        flat = child.reshape(z, weights.csn * weights.cwe)
        gathered = jnp.take(flat, child_idx, axis=1)
        gathered = gathered.reshape(z, weights.parent_lin.shape[0], weights.stencil)
        return jnp.mean(gathered, axis=2)
    if child.ndim == 2:
        flat = child.reshape(weights.csn * weights.cwe)
        gathered = jnp.take(flat, child_idx, axis=0)
        gathered = gathered.reshape(weights.parent_lin.shape[0], weights.stencil)
        return jnp.mean(gathered, axis=1)
    raise ValueError(f"feedback expects a 2D or 3D child field, got {child.shape}")


def feedback_to_parent_grid(child: jax.Array, weights: FeedbackWeights) -> jax.Array:
    """Return a parent-shaped feedback product, zero outside the overlap."""

    values = _feedback_values(child, weights)
    if child.ndim == 3:
        z = int(child.shape[0])
        out = jnp.zeros((z, weights.psn * weights.pwe), dtype=child.dtype)
        out = out.at[:, weights.parent_lin].set(values)
        return out.reshape(z, weights.psn, weights.pwe)
    out = jnp.zeros((weights.psn * weights.pwe,), dtype=child.dtype)
    out = out.at[weights.parent_lin].set(values)
    return out.reshape(weights.psn, weights.pwe)


def feedback_mask(weights: FeedbackWeights) -> jax.Array:
    """Parent-grid mask with one where feedback writes."""

    mask = jnp.zeros((weights.psn * weights.pwe,), dtype=jnp.float64)
    mask = mask.at[weights.parent_lin].set(1.0)
    return mask.reshape(weights.psn, weights.pwe)


def apply_feedback(
    parent: jax.Array,
    child: jax.Array,
    weights: FeedbackWeights,
    *,
    feedback: bool = True,
) -> jax.Array:
    """Write child-overlap averages onto the parent field.

    With ``feedback=False`` this returns ``parent`` unchanged, which is the
    baseline one-way nesting gate.
    """

    if not bool(feedback):
        return parent
    values = _feedback_values(child, weights)
    if parent.ndim == 3:
        z = int(parent.shape[0])
        flat = parent.reshape(z, weights.psn * weights.pwe)
        flat = flat.at[:, weights.parent_lin].set(values.astype(parent.dtype))
        return flat.reshape(z, weights.psn, weights.pwe)
    if parent.ndim == 2:
        flat = parent.reshape(weights.psn * weights.pwe)
        flat = flat.at[weights.parent_lin].set(values.astype(parent.dtype))
        return flat.reshape(weights.psn, weights.pwe)
    raise ValueError(f"apply_feedback expects a 2D or 3D parent field, got {parent.shape}")


def _base_pressure(state: State) -> jax.Array:
    return state.p_total - state.p_perturbation


def _base_geopotential(state: State) -> jax.Array:
    return state.ph_total - state.ph_perturbation


def _base_mu(state: State) -> jax.Array:
    return state.mu_total - state.mu_perturbation


def apply_state_feedback(
    parent: State,
    child: State,
    weights: StateFeedbackWeights,
    *,
    feedback: bool = True,
) -> State:
    """Apply optional two-way feedback to the current ``State`` contract.

    The default runtime passes ``feedback=False``.  When enabled, feedback updates
    the WRF forced prognostic set plus carried moisture/TKE number fields on mass
    points.  Parent base-state leaves are preserved; total aliases are rebuilt
    from the fed-back perturbation leaves.
    """

    if not bool(feedback):
        return parent

    mass = weights.mass
    u = apply_feedback(parent.u, child.u, weights.u, feedback=True)
    v = apply_feedback(parent.v, child.v, weights.v, feedback=True)
    w = apply_feedback(parent.w, child.w, mass, feedback=True)
    theta = apply_feedback(parent.theta, child.theta, mass, feedback=True)
    qv = apply_feedback(parent.qv, child.qv, mass, feedback=True)
    p_pert = apply_feedback(parent.p_perturbation, child.p_perturbation, mass, feedback=True)
    ph_pert = apply_feedback(parent.ph_perturbation, child.ph_perturbation, mass, feedback=True)
    mu_pert = apply_feedback(parent.mu_perturbation, child.mu_perturbation, mass, feedback=True)

    updates = {
        "u": u,
        "v": v,
        "w": w,
        "theta": theta,
        "qv": qv,
        "p_perturbation": p_pert,
        "p_total": _base_pressure(parent) + p_pert,
        "p": _base_pressure(parent) + p_pert,
        "ph_perturbation": ph_pert,
        "ph_total": _base_geopotential(parent) + ph_pert,
        "ph": _base_geopotential(parent) + ph_pert,
        "mu_perturbation": mu_pert,
        "mu_total": _base_mu(parent) + mu_pert,
        "mu": _base_mu(parent) + mu_pert,
    }
    for name in ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng", "qke", "Nc", "Nn"):
        if hasattr(parent, name) and hasattr(child, name):
            updates[name] = apply_feedback(getattr(parent, name), getattr(child, name), mass, feedback=True)
    return parent.replace(_cast=False, **updates)


def feedback_overlap_conservation(
    child: jax.Array,
    weights: FeedbackWeights,
    *,
    leaf: str,
    rel_tol: float = 1.0e-12,
) -> ConservationResult:
    """Check integral conservation over the feedback overlap."""

    values = _feedback_values(child, weights)
    child_idx = weights.child_lin.reshape(-1)
    if child.ndim == 3:
        z = int(child.shape[0])
        flat = child.reshape(z, weights.csn * weights.cwe)
        donors = jnp.take(flat, child_idx, axis=1).reshape(
            z, weights.parent_lin.shape[0], weights.stencil
        )
        child_sum = jnp.sum(donors)
        parent_sum = jnp.sum(values) * weights.stencil
    else:
        flat = child.reshape(weights.csn * weights.cwe)
        donors = jnp.take(flat, child_idx, axis=0).reshape(
            weights.parent_lin.shape[0], weights.stencil
        )
        child_sum = jnp.sum(donors)
        parent_sum = jnp.sum(values) * weights.stencil

    child_integral = float(child_sum)
    parent_integral = float(parent_sum)
    abs_residual = abs(parent_integral - child_integral)
    rel_residual = abs_residual / max(abs(child_integral), 1.0e-300)
    return ConservationResult(
        leaf=str(leaf),
        stagger=weights.stagger,
        n_cells=int(weights.parent_lin.shape[0]),
        child_overlap_integral=child_integral,
        parent_overlap_integral=parent_integral,
        abs_residual=abs_residual,
        rel_residual=rel_residual,
        conserved=bool(rel_residual <= float(rel_tol)),
    )


__all__ = [
    "ConservationResult",
    "FeedbackWeights",
    "StateFeedbackWeights",
    "apply_feedback",
    "apply_state_feedback",
    "build_feedback_weights",
    "build_state_feedback_weights",
    "feedback_mask",
    "feedback_overlap_conservation",
    "feedback_to_parent_grid",
]
