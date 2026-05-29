"""Device-side lateral-boundary forcing for the M6/ADR-023 d02 forecast driver.

This module reproduces the WRF v4 specified + relaxation-zone lateral boundary
update (``spec_bdytend`` / ``relax_bdytend`` in ``share/module_bc.F`` and the
weight table ``lbc_fcx_gcx`` in ``dyn_em/module_bc_em.F``) as a pure
``State -> State`` adapter, applied once per operational/replay timestep after
the dycore + physics block.

WRF subtleties faithfully reproduced here
------------------------------------------
* **Specified (outer) zone** of width ``spec_zone`` (the outermost row/column,
  ``b_dist=0``) is overwritten with the time-interpolated boundary value
  (WRF ``spec_bdytend`` drives the field tendency toward the boundary value; in
  the side-history replay we set the value directly).
* **Relaxation zone** of width ``relax_zone`` (rows ``b_dist = spec_zone ..
  relax_zone-1``) is nudged by
  ``field += fcx(b_dist)*fls0 - gcx(b_dist)*(fls1+fls2+fls3+fls4-4*fls0)``
  where ``fls0`` is the residual ``bdy - field`` at the row, ``fls1/fls2`` are
  the *tangential* residual neighbours (clamped at the side ends like WRF's
  ``max(i-1,ibs)``/``min(i+1,ibe)``), and ``fls3/fls4`` are the *normal*
  residual neighbours that use the boundary strips at width ``b_dist-1`` and
  ``b_dist+1`` against the adjacent interior rows -- the discrete Laplacian
  smoothing of the boundary residual, exactly as WRF computes it.
* **Corner trimming**: WRF's Y-boundary relaxation runs the tangential index
  ``i`` only over ``[b_limit+ibs , ibe-b_limit]`` with ``b_limit=b_dist`` so the
  diagonal corners are owned by the X (W/E) boundaries; the X-boundary
  relaxation trims its tangential index ``j`` to ``[b_dist+jbs+1 ,
  jbe-b_dist-1]``. We reproduce both via per-row tangential slicing so a
  relaxation-zone cell is updated by exactly one side, matching WRF.
* **Weights** ``fcx``/``gcx`` follow ``lbc_fcx_gcx`` for the
  ``specified``/``nested`` branches (identical linear taper; the optional
  ``spec_exp`` sponge multiplies both). We fold WRF's per-RK ``dt`` factor into
  the weights so a single per-step value update equals one WRF tendency step.

Known, documented departure from bit-exact WRF
----------------------------------------------
WRF relaxes the *mass-coupled* variables (``ru = c1*mu*u``, ``t = mu*theta``,
mass-weighted ``ph``/``w``). The Gen2 replay forces with *decoupled* wrfout
side-history (raw ``U``/``V``/``T``/...), so we relax the decoupled fields
against the decoupled boundary leaves. For the slowly-varying outer strip this
is an O(mu') approximation, not bit-exact WRF. It is the honest choice for a
side-history replay whose boundary data are themselves decoupled wrfout fields;
the residual it introduces is bounded by the column-mass perturbation and is
quantified in ``proofs/b4/``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp

from gpuwrf.contracts.state import State


SIDES = ("W", "E", "S", "N")
SIDE_INDEX = {name: index for index, name in enumerate(SIDES)}


@dataclass(frozen=True)
class BoundaryConfig:
    """WRF lateral-boundary control values for the pinned Gen2 d02 run.

    Mirrors the d01 ``namelist.input`` (``spec_bdy_width=5``, ``spec_zone=1``,
    ``relax_zone=4``, ``spec_exp=0``). ``update_cadence_s`` is the wrfbdy /
    side-history interval (hourly for the Gen2 corpus).
    """

    spec_bdy_width: int = 5
    spec_zone: int = 1
    relax_zone: int = 4
    update_cadence_s: float = 3600.0
    spec_exp: float = 0.0


DEFAULT_BOUNDARY_CONFIG = BoundaryConfig()


def apply_lateral_boundaries(
    state: State,
    lead_seconds,
    dt_s: float,
    config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
) -> State:
    """Apply the WRF specified outer zone + relaxation-zone nudging in-place.

    Reads the ``*_bdy`` leaves (shape ``(time, side, bdy_width, z, side_len)``)
    and the corresponding interior fields; writes ``u,v,w,theta,qv`` and the
    ``p/ph/mu`` total+perturbation triples in the boundary strip only. The
    interior beyond ``relax_zone`` is untouched. Field dtypes are preserved, so
    an fp64 (``force_fp64``) state stays fp64 and an operational fp32-gated
    state stays fp32-gated.
    """

    u = _apply_3d(state.u, state.u_bdy, lead_seconds, dt_s, config)
    v = _apply_3d(state.v, state.v_bdy, lead_seconds, dt_s, config)
    w = _apply_3d(state.w, state.w_bdy, lead_seconds, dt_s, config)
    theta = _apply_3d(state.theta, state.theta_bdy, lead_seconds, dt_s, config)
    qv = jnp.maximum(_apply_3d(state.qv, state.qv_bdy, lead_seconds, dt_s, config), 0.0)
    p_perturbation = _apply_3d(state.p_perturbation, state.p_bdy, lead_seconds, dt_s, config)
    pb = _apply_3d(_base_pressure(state), state.pb_bdy, lead_seconds, dt_s, config)
    ph_perturbation = _apply_3d(state.ph_perturbation, state.ph_bdy, lead_seconds, dt_s, config)
    phb = _apply_3d(_base_geopotential(state), state.phb_bdy, lead_seconds, dt_s, config)
    mu_perturbation = _apply_3d(state.mu_perturbation[None, :, :], state.mu_bdy, lead_seconds, dt_s, config)[0]
    mub = _apply_3d(_base_mu(state)[None, :, :], state.mub_bdy, lead_seconds, dt_s, config)[0]
    return state.replace(
        u=u,
        v=v,
        w=w,
        theta=theta,
        qv=qv,
        p_total=pb + p_perturbation,
        p_perturbation=p_perturbation,
        ph_total=phb + ph_perturbation,
        ph_perturbation=ph_perturbation,
        mu_total=mub + mu_perturbation,
        mu_perturbation=mu_perturbation,
    )


def interpolate_boundary_leaf(boundary, lead_seconds, cadence_s: float = 3600.0):
    """Linearly interpolate one boundary leaf along its leading time axis.

    ``boundary`` is ``(time, side, bdy_width, z, side_len)``; the result drops
    the time axis: ``(side, bdy_width, z, side_len)``. Equivalent to WRF's
    ``field_bdy + dtbc*field_bdy_tend`` linear-in-time forcing.
    """

    max_index = int(boundary.shape[0]) - 1
    lead_index = jnp.asarray(lead_seconds, dtype=jnp.float64) / float(cadence_s)
    lower = jnp.clip(jnp.floor(lead_index).astype(jnp.int32), 0, max_index)
    upper = jnp.clip(lower + 1, 0, max_index)
    alpha = jnp.clip(lead_index - lower.astype(jnp.float64), 0.0, 1.0)
    lower_values = jnp.take(boundary, lower, axis=0)
    upper_values = jnp.take(boundary, upper, axis=0)
    return (lower_values * (1.0 - alpha) + upper_values * alpha).astype(boundary.dtype)


def _base_pressure(state: State):
    return state.p_total - state.p_perturbation


def _base_geopotential(state: State):
    return state.ph_total - state.ph_perturbation


def _base_mu(state: State):
    return state.mu_total - state.mu_perturbation


def _apply_3d(field, boundary, lead_seconds, dt_s: float, config: BoundaryConfig):
    """WRF spec-zone + relax-zone update for one ``(z, y, x)`` field.

    ``forcing`` is the time-interpolated strip ``(side, bdy_width, z, side_len)``
    whose ``bdy_width`` axis runs outer (index 0, the domain edge) to inner.
    """

    forcing = interpolate_boundary_leaf(boundary, lead_seconds, config.update_cadence_s)
    # WRF applies spec_bdytend (outer zone) and relax_bdytend (relaxation zone)
    # in one pass.  WRF computes every side's relaxation tendency from the SAME
    # input field within an RK substep (the field is not mutated mid-pass), so
    # we evaluate all four relaxation slices against the original ``field`` and
    # only then scatter them.  With WRF corner trimming each relaxation cell is
    # owned by exactly one side, so the scatter order is immaterial.
    out = field
    for side in SIDES:
        out = _apply_side_relax(out, field, forcing, side, dt_s, config)
    # Overwrite the spec zone last so the outer specified rows are exactly the
    # boundary value (WRF spec zone is a hard set; relaxation never touches
    # b_dist < spec_zone).
    for side in SIDES:
        out = _apply_side_spec(out, forcing, side, config)
    return out


def _strip(forcing, side: str, width_index: int, z_len: int, side_len: int):
    """Return the boundary strip ``(z, side_len)`` for one side and bdy width.

    Two leaf layouts are accepted (per the frozen State docstring):
    * Current ``(side, bdy_width, z, side_len)`` -> per-side ``(bdy_width, z,
      side_len)``; we index the requested ``bdy_width``.
    * Legacy ``(side, z, side_len)`` (no ``bdy_width`` axis) -> per-side ``(z,
      side_len)``; every width maps to the same single strip (the relaxation
      normal-neighbour terms then collapse, which is the documented legacy
      behaviour -- only the spec zone is exact).

    ``side_len`` is the tangential extent (``y`` for W/E, ``x`` for S/N), padded
    to ``max(nx+1, ny+1)`` so we slice to the field's actual tangential length.
    """

    side_values = forcing[SIDE_INDEX[side]]
    if side_values.ndim == 3:  # (bdy_width, z, side_len)
        max_width = int(side_values.shape[0]) - 1
        width_index = min(max(int(width_index), 0), max_width)
        return side_values[width_index, :z_len, :side_len]
    # legacy (z, side_len): no bdy_width axis
    return side_values[:z_len, :side_len]


# ---------------------------------------------------------------------------
# Specified (outer) zone: hard-set the outermost spec_zone rows/cols.
# ---------------------------------------------------------------------------


def _apply_side_spec(field, forcing, side: str, config: BoundaryConfig):
    z_len, y_len, x_len = field.shape
    out = field
    for b_dist in range(int(config.spec_zone)):
        if side == "W":
            out = out.at[:, :, b_dist].set(_strip(forcing, side, b_dist, z_len, y_len))
        elif side == "E":
            out = out.at[:, :, x_len - 1 - b_dist].set(_strip(forcing, side, b_dist, z_len, y_len))
        elif side == "S":
            out = out.at[:, b_dist, :].set(_strip(forcing, side, b_dist, z_len, x_len))
        else:  # N
            out = out.at[:, y_len - 1 - b_dist, :].set(_strip(forcing, side, b_dist, z_len, x_len))
    return out


# ---------------------------------------------------------------------------
# Relaxation zone: WRF relax_bdytend stencil on the residual (bdy - field).
# ---------------------------------------------------------------------------


def _shift_clamp(values, shift: int, axis: int):
    """Shift ``values`` by ``shift`` along ``axis``, clamping (reflecting) the
    edge element to match WRF's ``max(i-1,ibs)`` / ``min(i+1,ibe)``."""

    n = values.shape[axis]
    if shift == 1:  # tangential s-1 neighbour: index max(i-1, 0)
        idx = jnp.concatenate((jnp.array([0]), jnp.arange(n - 1)))
    else:  # tangential s+1 neighbour: index min(i+1, n-1)
        idx = jnp.concatenate((jnp.arange(1, n), jnp.array([n - 1])))
    return jnp.take(values, idx, axis=axis)


def _relax_row(field, forcing, side, b_dist, z_len, side_len, *, weight_f, weight_g):
    """Compute the WRF-relaxed row ``b_dist`` for one side.

    Returns the full ``(z, side_len)`` relaxed slice; the caller masks it to the
    WRF tangential extent (corner trimming) before scattering.

    fls0 = bdy[b_dist]   - field(row b_dist)
    fls1 = bdy[b_dist]   - field(row b_dist), tangential +s neighbour
    fls2 = bdy[b_dist]   - field(row b_dist), tangential -s neighbour
    fls3 = bdy[b_dist-1] - field(row b_dist-1)            (normal, toward edge)
    fls4 = bdy[b_dist+1] - field(row b_dist+1)            (normal, toward interior)
    """

    if side == "W":
        cur = field[:, :, b_dist]
        toward_edge = field[:, :, b_dist - 1]
        toward_interior = field[:, :, b_dist + 1]
    elif side == "E":
        x = field.shape[2] - 1 - b_dist
        cur = field[:, :, x]
        toward_edge = field[:, :, x + 1]
        toward_interior = field[:, :, x - 1]
    elif side == "S":
        cur = field[:, b_dist, :]
        toward_edge = field[:, b_dist - 1, :]
        toward_interior = field[:, b_dist + 1, :]
    else:  # N
        y = field.shape[1] - 1 - b_dist
        cur = field[:, y, :]
        toward_edge = field[:, y + 1, :]
        toward_interior = field[:, y - 1, :]

    bdy0 = _strip(forcing, side, b_dist, z_len, side_len)
    bdy_edge = _strip(forcing, side, b_dist - 1, z_len, side_len)
    bdy_interior = _strip(forcing, side, b_dist + 1, z_len, side_len)

    fls0 = bdy0 - cur
    # tangential axis of the (z, side_len) slice is axis 1
    fls1 = _shift_clamp(fls0, +1, axis=1)
    fls2 = _shift_clamp(fls0, -1, axis=1)
    fls3 = bdy_edge - toward_edge
    fls4 = bdy_interior - toward_interior
    laplacian = fls1 + fls2 + fls3 + fls4 - 4.0 * fls0
    return cur + weight_f * fls0 - weight_g * laplacian


def _apply_side_relax(out, field, forcing, side: str, dt_s: float, config: BoundaryConfig):
    """Scatter one side's relaxation rows into ``out``; stencil reads ``field``.

    ``field`` is the unmodified input (WRF reads the same field for all sides in
    one RK substep); ``out`` is the accumulating result that we write into.
    """

    z_len, y_len, x_len = field.shape
    spec_zone = int(config.spec_zone)
    relax_zone = int(config.relax_zone)
    for b_dist in range(spec_zone, relax_zone):
        weight_f, weight_g = _wrf_relax_weights(b_dist, dt_s, config)
        if side in ("W", "E"):
            side_len = y_len
            relaxed = _relax_row(field, forcing, side, b_dist, z_len, side_len, weight_f=weight_f, weight_g=weight_g)
            # WRF X-boundary relaxation tangential j in [b_dist+jbs+1, jbe-b_dist-1]
            start, end = b_dist + 1, y_len - b_dist - 1
            col = b_dist if side == "W" else x_len - 1 - b_dist
            out = out.at[:, start:end, col].set(relaxed[:, start:end])
        else:  # S, N
            side_len = x_len
            relaxed = _relax_row(field, forcing, side, b_dist, z_len, side_len, weight_f=weight_f, weight_g=weight_g)
            # WRF Y-boundary relaxation tangential i in [b_limit+ibs, ibe-b_limit]
            # with b_limit=b_dist -> corners owned by W/E so the diagonal is
            # updated exactly once.
            start, end = b_dist, x_len - b_dist
            row = b_dist if side == "S" else y_len - 1 - b_dist
            out = out.at[:, row, start:end].set(relaxed[:, start:end])
    return out


def _wrf_relax_weights(b_dist: int, dt_s: float, config: BoundaryConfig) -> tuple[float, float]:
    """Return ``dt*fcx`` and ``dt*gcx`` for relaxation row ``b_dist``.

    Reproduces ``lbc_fcx_gcx`` (``dyn_em/module_bc_em.F``). WRF indexes the
    weight by Fortran ``loop = b_dist + 1`` (1-based). With ``spec_zone=1``,
    ``relax_zone=4`` the active rows are ``b_dist = 1,2,3`` -> ``loop = 2,3,4``
    -> linear taper ``1, 2/3, 1/3``. The optional ``spec_exp`` sponge is the
    ``exp(-(loop-(spec_zone+1))*spec_exp)`` multiplier (0 for the pinned run).

    The per-step value update folds WRF's RK ``dt`` factor in: WRF accumulates a
    tendency ``fcx*fls0 - ...`` integrated over ``dt``; one decoupled value step
    is ``dt*fcx*fls0 - ...``.
    """

    loop_1based = int(b_dist) + 1
    numerator = float(config.spec_zone + config.relax_zone - loop_1based)
    denominator = float(config.relax_zone - 1)
    linear = max(0.0, numerator / denominator) if denominator > 0.0 else 0.0
    sponge = math.exp(-(loop_1based - (config.spec_zone + 1)) * config.spec_exp)
    fcx = 0.1 / float(dt_s) * linear * sponge
    gcx = 1.0 / float(dt_s) / 50.0 * linear * sponge
    return float(dt_s) * fcx, float(dt_s) * gcx


__all__ = [
    "BoundaryConfig",
    "DEFAULT_BOUNDARY_CONFIG",
    "SIDES",
    "SIDE_INDEX",
    "apply_lateral_boundaries",
    "interpolate_boundary_leaf",
]
