"""GPUWRF_W_SURFACE_RESET (b-core LL01 Finding 5, default off): WRF's end-of-step kinematic surface w.

WRF ``solve_em.F:4818-4834`` ends EVERY time step -- after all RK stages, the microphysics and the
lateral boundary updates -- with ``set_w_surface(fill_w_flag=.false.)`` (``module_bc_em.F:1196-1295``):
w at the lowest face (k=1) is reset from the FINAL u_2/v_2, the terrain height and the cf1/cf2/cf3
surface extrapolation, with edge-clamped neighbours (jm1 = max(j-1, jds), jp1 = min(j+1, jde-1), same
in i; periodic_x / periodic_y wrap through the WRF halo). Without it w(k=0) stays the small_step_finish recovery from the acoustic
perturbation and is never re-anchored to the final winds: CPU-WRF history W(k=0) equals this formula to
round-off (0227 d03 rms 7e-8 m/s), the v0.3.1 GPU twin differed by up to 0.16 m/s on slopes.
"""
import os

import numpy as np

ENABLED = os.environ.get("GPUWRF_W_SURFACE_RESET", "0") == "1"


def kinematic_surface_w(u, v, ht, msftx, msfty, cf1, cf2, cf3, *, dx, dy, periodic_x=False, periodic_y=False,
                        dtype=None):
    """WRF ``set_w_surface`` lower-boundary w(k=1) on the mass grid (ny, nx), in ``dtype`` arithmetic.

    ``u`` (nz, ny, nx+1), ``v`` (nz, ny+1, nx), ``ht``/``msftx``/``msfty`` (ny, nx); the expression and its
    evaluation order are WRF's (left to right, REAL literals and ``rdx = 1./dx`` rounded in ``dtype``)."""
    import jax.numpy as jnp

    dt = np.dtype(dtype if dtype is not None else u.dtype)
    one = np.asarray(1.0, dt)
    rdx = one / np.asarray(dx, dt)
    rdy = one / np.asarray(dy, dt)
    half = np.asarray(0.5, dt)
    u, v, ht, msftx, msfty = (jnp.asarray(a).astype(dt) for a in (u, v, ht, msftx, msfty))
    c1, c2, c3 = (jnp.asarray(c).astype(dt).reshape(()) for c in (cf1, cf2, cf3))
    vv = c1 * v[0] + c2 * v[1] + c3 * v[2]  # cf1*v(1)+cf2*v(2)+cf3*v(3) on the v rows (ny+1, nx)
    uu = c1 * u[0] + c2 * u[1] + c3 * u[2]  # same on the u columns (ny, nx+1)
    if periodic_y:  # jm1_limit = jds-1, jp1_limit = jde: the periodic halo row
        ht_jp1 = jnp.roll(ht, -1, axis=0)
        ht_jm1 = jnp.roll(ht, 1, axis=0)
    else:
        ht_jp1 = jnp.concatenate([ht[1:], ht[-1:]], axis=0)  # jp1 = min(j+1, jde-1)
        ht_jm1 = jnp.concatenate([ht[:1], ht[:-1]], axis=0)  # jm1 = max(j-1, jds)
    if periodic_x:
        ht_ip1 = jnp.roll(ht, -1, axis=1)
        ht_im1 = jnp.roll(ht, 1, axis=1)
    else:
        ht_ip1 = jnp.concatenate([ht[:, 1:], ht[:, -1:]], axis=1)
        ht_im1 = jnp.concatenate([ht[:, :1], ht[:, :-1]], axis=1)
    return (
        msfty * half * rdy * ((ht_jp1 - ht) * vv[1:] + (ht - ht_jm1) * vv[:-1])
        + msftx * half * rdx * ((ht_ip1 - ht) * uu[:, 1:] + (ht - ht_im1) * uu[:, :-1])
    )


def reset_surface_w(state, grid, metrics, *, periodic_x=False, periodic_y=False):
    """``state.w`` with face 0 replaced by the kinematic value of the state's own (final) u/v.

    The operational port has no periodic-y lateral condition (idealized runs are periodic in x only,
    ``_acoustic_lateral_bc_flags``), so its caller passes ``periodic_y=False``."""
    w = state.w
    surface = kinematic_surface_w(
        state.u, state.v, grid.terrain_height, metrics.msftx, metrics.msfty,
        metrics.cf1, metrics.cf2, metrics.cf3,
        dx=float(grid.projection.dx_m), dy=float(grid.projection.dy_m),
        periodic_x=periodic_x, periodic_y=periodic_y, dtype=w.dtype,
    )
    return w.at[0].set(surface)
