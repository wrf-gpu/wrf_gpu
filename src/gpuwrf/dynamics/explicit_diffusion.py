"""Explicit diffusion tendencies for the dry dycore (Block 1 stabiliser).

Two WRF-faithful paths, both returned as *uncoupled* tendencies (du/dt etc.) so
they add into the operational RK tendency convention (the roll-advection path
also produces uncoupled tendencies):

1. ``sixth_order_diffusion_tendency`` — WRF ``sixth_order_diffusion``
   (``module_big_step_utilities_em.F:6504-6920``) with the monotonic flux
   limiter (``diff_6th_opt=2``).  This is the operational d02 numerical filter
   (``diff_6th_opt=2``, ``diff_6th_factor=0.12``) that suppresses 2dx noise.

2. ``constant_k_diffusion_tendency`` — WRF ``horizontal_diffusion`` /
   ``vertical_diffusion`` with a constant eddy viscosity ``khdif=kvdif=K``
   (``module_big_step_utilities_em.F:2999-3234``).  The Straka et al. (1993)
   density-current reference solution is *defined* with constant ν = 75 m²/s on
   u, v, θ, so this is part of the test definition, not a masking clamp.

Periodic-x/-y only (the idealized + audit configuration).  The original slab
helpers keep unit-map-factor defaults; the WRF terrain-following deformation
helpers below accept real map factors and ``zx``/``zy`` slope metrics.
"""

from __future__ import annotations

import jax
from jax import config
import jax.numpy as jnp


config.update("jax_enable_x64", True)


GRAVITY_M_S2 = 9.81


def _core_x(field: jax.Array, nx: int) -> jax.Array:
    """Drop a periodic x-wrap point when present."""

    return field[..., :nx] if field.shape[-1] == nx + 1 else field


def _core_y(field: jax.Array, ny: int) -> jax.Array:
    """Drop a periodic y-wrap point when present."""

    return field[..., :ny, :] if field.shape[-2] == ny + 1 else field


def _mass_metric_or_one(metric: jax.Array | None, reference: jax.Array) -> jax.Array:
    """Return a mass-grid metric, defaulting to one on ``reference``'s y/x shape."""

    if metric is None:
        return jnp.ones(reference.shape[-2:], dtype=reference.dtype)
    return jnp.asarray(metric, dtype=reference.dtype)


def _x_metric_or_one(metric: jax.Array | None, *, nx: int, reference: jax.Array) -> jax.Array:
    """Return an x-face metric without the periodic wrap point."""

    if metric is None:
        return jnp.ones(reference.shape[-2:-1] + (nx,), dtype=reference.dtype)
    return _core_x(jnp.asarray(metric, dtype=reference.dtype), nx)


def _y_metric_or_one(metric: jax.Array | None, *, ny: int, reference: jax.Array) -> jax.Array:
    """Return a y-face metric without the periodic wrap point."""

    if metric is None:
        return jnp.ones((ny,) + reference.shape[-1:], dtype=reference.dtype)
    return _core_y(jnp.asarray(metric, dtype=reference.dtype), ny)


def _zface_or_zero(metric: jax.Array | None, *, nz: int, ny: int, nx: int, reference: jax.Array) -> jax.Array:
    """Return a ``(nz+1, ny, nx)`` z-face metric, defaulting to zero."""

    if metric is None:
        return jnp.zeros((nz + 1, ny, nx), dtype=reference.dtype)
    arr = jnp.asarray(metric, dtype=reference.dtype)
    if arr.shape[0] == nz:
        arr = jnp.concatenate((arr, arr[-1:, :, :]), axis=0)
    return arr


def _mass3_or_one(metric: jax.Array | None, reference: jax.Array) -> jax.Array:
    """Return a 3-D mass-level metric, defaulting to one."""

    if metric is None:
        return jnp.ones_like(reference)
    return jnp.asarray(metric, dtype=reference.dtype)


def _wface3_or_one(metric: jax.Array | None, *, nz: int, ny: int, nx: int, reference: jax.Array) -> jax.Array:
    """Return a ``(nz+1, ny, nx)`` w-face metric, defaulting to one."""

    if metric is None:
        return jnp.ones((nz + 1, ny, nx), dtype=reference.dtype)
    arr = jnp.asarray(metric, dtype=reference.dtype)
    if arr.shape[0] == nz:
        arr = jnp.concatenate((arr[:1, :, :], arr), axis=0)
    return arr


def _vertical_face_average_pair(
    left: jax.Array,
    right: jax.Array,
    *,
    fnm: jax.Array | None = None,
    fnp: jax.Array | None = None,
    cf1: jax.Array | float | None = None,
    cf2: jax.Array | float | None = None,
    cf3: jax.Array | float | None = None,
    dn: jax.Array | None = None,
    dnw: jax.Array | None = None,
) -> jax.Array:
    """WRF z-face average/extrapolation of two mass-level columns.

    Used by ``cal_deform_and_div`` before multiplying by ``zx``/``zy``.  Interior
    faces use ``fnm/fnp``; bottom uses ``cf1/cf2/cf3``; top uses WRF's linear
    extrapolation coefficient ``cft2=-0.5*dnw(kte-1)/dn(kte-1)``.
    """

    nz = int(left.shape[0])
    dtype = left.dtype
    pair = 0.5 * (left + right)
    face = jnp.zeros((nz + 1,) + tuple(left.shape[1:]), dtype=dtype)
    if nz > 1:
        if fnm is None:
            fnm_v = jnp.ones((nz,), dtype=dtype)
            fnp_v = jnp.zeros((nz,), dtype=dtype)
        else:
            fnm_v = jnp.asarray(fnm, dtype=dtype)
            fnp_v = jnp.asarray(fnp, dtype=dtype) if fnp is not None else jnp.zeros((nz,), dtype=dtype)
        interior = 0.5 * (
            fnm_v[1:nz, None, None] * (left[1:] + right[1:])
            + fnp_v[1:nz, None, None] * (left[:-1] + right[:-1])
        )
        face = face.at[1:nz, :, :].set(interior)
    if nz >= 3 and cf1 is not None and cf2 is not None and cf3 is not None:
        bottom = 0.5 * (
            jnp.asarray(cf1, dtype=dtype) * (left[0] + right[0])
            + jnp.asarray(cf2, dtype=dtype) * (left[1] + right[1])
            + jnp.asarray(cf3, dtype=dtype) * (left[2] + right[2])
        )
    else:
        bottom = pair[0]
    face = face.at[0, :, :].set(bottom)
    if nz >= 2 and dn is not None and dnw is not None:
        dn_v = jnp.asarray(dn, dtype=dtype)
        dnw_v = jnp.asarray(dnw, dtype=dtype)
        cft2 = -0.5 * dnw_v[nz - 1] / dn_v[nz - 1]
        cft1 = 1.0 - cft2
        top = 0.5 * (cft1 * (left[nz - 1] + right[nz - 1]) + cft2 * (left[nz - 2] + right[nz - 2]))
    else:
        top = pair[nz - 1]
    face = face.at[nz, :, :].set(top)
    return face


def _dflux6(field: jax.Array, axis: int) -> tuple[jax.Array, jax.Array]:
    """WRF 6th-order diffusive flux pair (Xue eq. 3) at faces p0 (i) and p1 (i+1).

    ``dflux_p0 = 10*(f(i)-f(i-1)) - 5*(f(i+1)-f(i-2)) + (f(i+2)-f(i-3))`` located
    at the left face of cell ``i``; ``dflux_p1`` is the same shifted to ``i+1``.
    Returns ``(dflux_p0, dflux_p1, grad_p0, grad_p1)`` where the gradients
    ``f(i)-f(i-1)`` / ``f(i+1)-f(i)`` are used by the monotonic limiter.
    """

    fm3 = jnp.roll(field, 3, axis=axis)
    fm2 = jnp.roll(field, 2, axis=axis)
    fm1 = jnp.roll(field, 1, axis=axis)
    f0 = field
    fp1 = jnp.roll(field, -1, axis=axis)
    fp2 = jnp.roll(field, -2, axis=axis)
    fp3 = jnp.roll(field, -3, axis=axis)
    dflux_p0 = 10.0 * (f0 - fm1) - 5.0 * (fp1 - fm2) + (fp2 - fm3)
    dflux_p1 = 10.0 * (fp1 - f0) - 5.0 * (fp2 - fm1) + (fp3 - fm2)
    grad_p0 = f0 - fm1
    grad_p1 = fp1 - f0
    return dflux_p0, dflux_p1, grad_p0, grad_p1


def _sixth_axis(field: jax.Array, axis: int, coef: float, monotonic: bool) -> jax.Array:
    """Uncoupled 6th-order diffusion tendency along one axis (periodic).

    WRF coupled form: ``tend = coef*(mu_p1*dflux_p1 - mu_p0*dflux_p0)``, and the
    prognostic later divides by mass; with unit map factors and the perturbation
    update dividing by the same mass, the *uncoupled* tendency is
    ``coef*(dflux_p1 - dflux_p0)`` (mu cancels to first order for the column).
    """

    dflux_p0, dflux_p1, grad_p0, grad_p1 = _dflux6(field, axis)
    if monotonic:
        # diff_6th_opt=2: prohibit up-gradient diffusion (Xue eq. 10 variant).
        dflux_p0 = jnp.where(dflux_p0 * grad_p0 <= 0.0, 0.0, dflux_p0)
        dflux_p1 = jnp.where(dflux_p1 * grad_p1 <= 0.0, 0.0, dflux_p1)
    return float(coef) * (dflux_p1 - dflux_p0)


def sixth_order_diffusion_tendency(
    field: jax.Array,
    *,
    dt: float,
    diff_6th_factor: float,
    horizontal_only: bool = True,
    monotonic: bool = True,
) -> jax.Array:
    """Return the WRF 6th-order numerical-diffusion tendency for one 3-D field.

    Source: WRF ``module_big_step_utilities_em.F:6504-6920``.  The coefficient is
    ``diff_6th_coef = diff_6th_factor * 0.015625 / (2*dt)`` (``:6605``).  WRF
    applies the filter on the horizontal coordinate surfaces (x and y); the
    one-row idealized slab and the audit case are effectively x-only in the
    horizontal, with the y-axis a singleton (its roll-stencil contribution is
    zero on a 1-wide axis).
    """

    coef = float(diff_6th_factor) * 0.015625 / (2.0 * float(dt))
    tend = _sixth_axis(field, axis=2, coef=coef, monotonic=monotonic)
    if field.shape[1] > 1:
        tend = tend + _sixth_axis(field, axis=1, coef=coef, monotonic=monotonic)
    if not horizontal_only and field.shape[0] > 1:
        # WRF applies the 6th-order filter only on coordinate (horizontal)
        # surfaces; vertical 6th-order is not part of diff_6th_opt.  Kept off.
        pass
    return tend


def _laplacian_axis_periodic(field: jax.Array, axis: int, spacing: float) -> jax.Array:
    """Second-order periodic Laplacian d2f/dx2 along ``axis``."""

    return (
        jnp.roll(field, -1, axis=axis) - 2.0 * field + jnp.roll(field, 1, axis=axis)
    ) / (float(spacing) * float(spacing))


def _laplacian_z_rigid(field: jax.Array, spacing) -> jax.Array:
    """Second-order vertical Laplacian with zero-gradient (rigid) top/bottom.

    ``spacing`` may be a Python float or a traced JAX scalar (used inside jit).
    """

    nz = int(field.shape[0])
    if nz < 3:
        return jnp.zeros_like(field)
    sp2 = jnp.asarray(spacing, dtype=field.dtype) ** 2
    interior = (field[2:, :, :] - 2.0 * field[1:-1, :, :] + field[:-2, :, :]) / sp2
    lap = jnp.zeros_like(field)
    lap = lap.at[1:-1, :, :].set(interior)
    return lap


def constant_k_diffusion_tendency(
    field: jax.Array,
    *,
    k_m2_s: float,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    horizontal: bool = True,
    vertical: bool = True,
) -> jax.Array:
    """Return a constant-viscosity (``K``) 2nd-order diffusion tendency.

    ``du/dt += K * (d2/dx2 + d2/dy2 + d2/dz2) field``.  Source: WRF
    ``horizontal_diffusion`` / ``vertical_diffusion`` with constant ``xkmhd=K``
    (``module_big_step_utilities_em.F:2999-3234``).  This is the Straka et al.
    (1993) ν = 75 m²/s definition (the reference solution is *defined* with it).
    Periodic horizontal, rigid vertical boundaries.
    """

    tend = jnp.zeros_like(field)
    if horizontal:
        tend = tend + float(k_m2_s) * _laplacian_axis_periodic(field, axis=2, spacing=dx_m)
        if field.shape[1] > 1:
            tend = tend + float(k_m2_s) * _laplacian_axis_periodic(field, axis=1, spacing=dy_m)
    if vertical and field.shape[0] > 1:
        tend = tend + float(k_m2_s) * _laplacian_z_rigid(field, spacing=dz_m)
    return tend


def conservative_constant_k_diffusion_tendency(
    field: jax.Array,
    *,
    mass: jax.Array,
    k_m2_s: float,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    horizontal: bool = True,
    vertical: bool = True,
) -> jax.Array:
    """Mass-conservative constant-K diffusion: ``d/dx_j( mass*K*d field/dx_j )``.

    Source: WRF ``horizontal_diffusion_s`` / ``vertical_diffusion`` (km_opt=1,
    ``module_diffusion_em.F:2999-3018, 3234``), which build the diffusive flux
    ``F_j = rho*xkh*d field/dx_j`` at cell faces and take its divergence so the
    column-mass-weighted integral of ``field`` is conserved to round-off.  The
    previous JAX path used the *non-conservative* ``mass*K*∇²field`` form, which
    leaks the dry-column mass integral at the sharp Straka cold front (relative
    drift ~3.4e-8 over 900 s; F7N).  This flux-divergence form is the WRF-faithful
    replacement: it returns the ALREADY mass-coupled tendency (do NOT multiply by
    ``mass`` again).  ``mass`` is the field-specific face/level dry-air mass
    ``c1*mu+c2`` (mass_h for theta, mass_u/mass_v/mass_f for u/v/w).  Periodic in
    the horizontal, rigid (zero-flux) top/bottom.
    """

    K = float(k_m2_s)
    tend = jnp.zeros_like(field)
    if horizontal:
        # x flux at face i+1/2: mass_face * K * (f(i+1)-f(i))/dx ; mass at face =
        # 0.5*(mass(i)+mass(i+1)).  div = (F(i+1/2)-F(i-1/2))/dx (periodic).
        mass_xf = 0.5 * (mass + jnp.roll(mass, -1, axis=2))
        fx = mass_xf * K * (jnp.roll(field, -1, axis=2) - field) / float(dx_m)
        tend = tend + (fx - jnp.roll(fx, 1, axis=2)) / float(dx_m)
        if field.shape[1] > 1:
            mass_yf = 0.5 * (mass + jnp.roll(mass, -1, axis=1))
            fy = mass_yf * K * (jnp.roll(field, -1, axis=1) - field) / float(dy_m)
            tend = tend + (fy - jnp.roll(fy, 1, axis=1)) / float(dy_m)
    if vertical and field.shape[0] > 2:
        # vertical flux at interior faces k+1/2 (between mass levels k and k+1);
        # zero flux through the rigid top/bottom -> conserves the column integral.
        sp = jnp.asarray(dz_m, dtype=field.dtype)
        mass_zf = 0.5 * (mass[:-1, :, :] + mass[1:, :, :])
        fz_int = mass_zf * K * (field[1:, :, :] - field[:-1, :, :]) / sp  # (nz-1,..) at faces 1..nz-1
        nz = int(field.shape[0])
        flux = jnp.zeros((nz + 1,) + tuple(field.shape[1:]), dtype=field.dtype)
        flux = flux.at[1:nz, :, :].set(fz_int)
        # divergence on mass levels: (flux(k+1/2)-flux(k-1/2))/dz
        tend = tend + (flux[1:, :, :] - flux[:nz, :, :]) / sp
    return tend


def _ddx_periodic(field: jax.Array, spacing: float, axis: int = 2) -> jax.Array:
    """Centered first derivative d/dx along a periodic axis (2nd order)."""

    return (jnp.roll(field, -1, axis=axis) - jnp.roll(field, 1, axis=axis)) / (2.0 * float(spacing))


def constant_k_deformation_momentum_tendency(
    u: jax.Array,
    w: jax.Array,
    *,
    k_m2_s: float,
    dx_m: float,
    dz_m: float,
) -> tuple[jax.Array, jax.Array]:
    """WRF diff_opt=2 / km_opt=1 constant-K *deformation-tensor* momentum diffusion.

    Returns ``(du/dt, dw/dt)`` uncoupled momentum-diffusion tendencies for the
    flat (zx=zy=0), uniform-z, periodic-x, single-row (ny=1) idealized slab.

    WRF (``module_diffusion_em.F``) diffuses momentum with the deformation
    stress tensor, NOT a plain Laplacian.  On the flat slab the relevant
    deformations reduce to (``cal_deform_and_div`` :41-47):
      * ``D11 = 2 du/dx``      (``:207-215``)
      * ``D33 = 2 dw/dz``      (``:368-373``)
      * ``D13 = dw/dx + du/dz`` (``:799-902``, vorticity points)
    and the stress is ``tau_ij = -K * rho * D_ij`` (``cal_titau_* :5428/:5714``,
    rho folds into the mass coupling done by the caller).  The momentum tendency
    is the stress divergence:
      * u :  d/dx(K D11) + d/dz(K D13) = 2K u_xx + K(u_zz + w_xz)
             (``horizontal_diffusion_u_2`` D11 path + ``vertical_diffusion_u_2``
             D13 path ``:4463-4571``)
      * w :  d/dx(K D13) + d/dz(K D33) = K(w_xx + u_xz) + 2K w_zz
             (``horizontal_diffusion_w_2`` :3519/:3695 D13 path +
             ``vertical_diffusion_w_2`` :4688/:4779 D33 path)

    vs the previous plain ``K(u_xx+u_zz)`` / ``K(w_xx+w_zz)``: the diagonal
    terms carry the WRF factor 2 (D11/D33), and the **cross terms** ``K w_xz``
    (in u) and ``K u_xz`` (in w) were entirely missing — these are first-order
    at the sheared descending cold front (large du/dz + dw/dx) and are what
    bound the head |w| and let the front mix/propagate (WRF ground truth
    ``proofs/m9/wrf_em_grav2d_x_front_*``: max|w| saturates ~22 m/s, theta'min
    decays -16.6 -> -11 K by 360 s; the plain-Laplacian JAX path leaves the
    front unmixed and the head |w| runs away to NaN ~270-300 s).

    ``u`` is on x-faces ``(nz, ny, nx+1)``; ``w`` on z-faces ``(nz+1, ny, nx)``.
    The returned ``du/dt`` is on the u stagger, ``dw/dt`` on the w stagger.
    """

    k = float(k_m2_s)
    dx = float(dx_m)
    # dz may be a traced JAX scalar (mean physical dz computed inside jit).
    dz = jnp.asarray(dz_m, dtype=u.dtype)
    nx_u = u.shape[-1]
    # work on the nx periodic mass columns of u (face i == mass-cell left face)
    u_f = u[:, :, : nx_u - 1] if nx_u > w.shape[-1] else u  # (nz, ny, nx)
    nz = w.shape[0] - 1

    # --- diagonal terms (factor 2 from D11=2 du/dx, D33=2 dw/dz) ---
    # u: 2K d2u/dx2 ; w: 2K d2w/dz2.  (vs plain Laplacian K d2/d.. -> factor 2.)
    u_xx = (jnp.roll(u_f, -1, axis=2) - 2.0 * u_f + jnp.roll(u_f, 1, axis=2)) / (dx * dx)
    du = 2.0 * k * u_xx
    # u vertical diagonal part K d2u/dz2 (from D13 = du/dz + dw/dx, the du/dz part)
    u_zz = jnp.zeros_like(u_f)
    if u_f.shape[0] >= 3:
        u_zz = u_zz.at[1:-1, :, :].set((u_f[2:, :, :] - 2.0 * u_f[1:-1, :, :] + u_f[:-2, :, :]) / (dz * dz))
    du = du + k * u_zz
    # w: K d2w/dx2 (from D13 = dw/dx + du/dz, the dw/dx part) + 2K d2w/dz2 (D33)
    w_xx = (jnp.roll(w, -1, axis=2) - 2.0 * w + jnp.roll(w, 1, axis=2)) / (dx * dx)
    w_zz = jnp.zeros_like(w)
    if nz >= 2:
        w_zz = w_zz.at[1:nz, :, :].set((w[2 : nz + 1, :, :] - 2.0 * w[1:nz, :, :] + w[0 : nz - 1, :, :]) / (dz * dz))
    dw = k * w_xx + 2.0 * k * w_zz

    # --- cross terms: u gets K d/dz(dw/dx) = K w_xz ; w gets K d/dx(du/dz) = K u_xz ---
    # w_xz on the u stagger: w is at mass-x / z-faces; dw/dx -> u-faces, then d/dz to
    # mass levels (where u lives).  Build dw/dx at mass-x by centered x-derivative of w
    # (on w faces), interpolate the z-derivative to mass levels.
    wx = _ddx_periodic(w, dx, axis=2)  # (nz+1, ny, nx) dw/dx at w faces (mass-x)
    # d/dz of wx from w-faces to mass levels: (wx(k+1)-wx(k))/dz  -> (nz, ny, nx)
    wxz_mass = (wx[1 : nz + 1, :, :] - wx[0:nz, :, :]) / dz  # (nz, ny, nx) at mass levels
    du = du + k * wxz_mass  # u_zz/u lives on mass levels; same stagger in z, mass-x ~ u-face approx
    # u_xz on the w stagger: du/dz at mass-x / z-faces, then d/dx.
    # du/dz from u-faces(mass-levels) to z-faces: (u(k)-u(k-1))/dz on faces 1..nz-1.
    uz_face = jnp.zeros_like(w)
    if nz >= 2:
        uz_face = uz_face.at[1:nz, :, :].set((u_f[1:nz, :, :] - u_f[0 : nz - 1, :, :]) / dz)
    uxz = _ddx_periodic(uz_face, dx, axis=2)  # d/dx of du/dz at w faces (mass-x)
    dw = dw + k * uxz

    # restore u to the (nx+1) face layout if needed (periodic wrap)
    if nx_u > w.shape[-1]:
        du = jnp.concatenate((du, du[:, :, :1]), axis=2)
    return du, dw


def wrf_deformation_momentum_tendency(
    u: jax.Array,
    w: jax.Array,
    *,
    rho: jax.Array,
    k_m2_s: float,
    dx_m: float,
    dz_m: jax.Array | float,
) -> tuple[jax.Array, jax.Array]:
    """WRF ``diff_opt=2`` deformation-tensor momentum diffusion (coupled tendency).

    Returns ``(d(rho*u)/dt, d(rho*w)/dt)`` -- the mass/density-weighted
    (*coupled*) momentum-diffusion tendencies, exactly as WRF's
    ``horizontal_diffusion_{u,w}_2`` + ``vertical_diffusion_{u,w}_2`` build the
    coupled ``ru_tendf``/``rw_tendf`` (``module_diffusion_em.F:2979-3007,
    4131-4155``).  This is the WRF momentum operator (stress divergence of the
    deformation tensor), NOT the scalar Laplacian / scalar flux divergence.

    Flat-slab reduction (zx=zy=0, msf=1, ny=1, uniform physical dz):

    * deformations (``cal_deform_and_div`` ``:215,:373,:820/:885``):
        - ``D11 = 2 du/dx``                (mass levels, u-x faces)
        - ``D33 = 2 dw/dz``                (mass levels, from w faces)
        - ``D13 = dw/dx + du/dz``          (w faces / vorticity points)
    * stresses ``titau_ij = -rho * K * D_ij`` (``cal_titau_* :5428,:5441``).
    * u-tendency (``horizontal_diffusion_u_2 :3308`` flat: ``g*dz/dnw * rdx *
      d/dx(titau11)``; ``vertical_diffusion_u_2 :4552`` flat:
      ``+ g/dnw * d/dz(titau13)``).
    * w-tendency (``horizontal_diffusion_w_2`` flat: ``g*dz/dn * rdx *
      d/dx(titau13)``; ``vertical_diffusion_w_2 :4779``:
      ``+ g * d/dz(titau33) / dn``).

    On the uniform-z neutral slab the mass-coordinate weights ``g*dz/dnw`` and
    ``g/dnw`` are the SAME constant column factor ``C = rho*K`` carries through
    as the only field-dependent term, so the operator reduces to the
    density-weighted stress divergence

        d(rho*u)/dt = d/dx( rho*K*D11 ) + d/dz( rho*K*D13 )
                    = d/dx( rho*K*2 u_x ) + d/dz( rho*K*(w_x + u_z) )
        d(rho*w)/dt = d/dx( rho*K*D13 ) + d/dz( rho*K*D33 )
                    = d/dx( rho*K*(w_x + u_z) ) + d/dz( rho*K*2 w_z )

    which is the mass-conservative (flux-divergence) WRF deformation operator
    with the diagonal factor 2 (D11/D33) and the off-diagonal cross terms
    (the ``w_x`` term in u and the ``u_z`` term in w) that the scalar Laplacian
    and the scalar flux-divergence both omit.  ``rho`` is the full air density
    (WRF ``grid%rho``, = inverse specific volume) on mass levels ``(nz,ny,nx)``.

    ``u`` is on x-faces ``(nz,ny,nx+1)`` (periodic: face nx == face 0); ``w`` on
    z-faces ``(nz+1,ny,nx)``.  Periodic x, rigid (zero-stress) top/bottom.
    """

    k = float(k_m2_s)
    dx = float(dx_m)
    dz = jnp.asarray(dz_m, dtype=u.dtype)
    nx = w.shape[-1]
    nz = w.shape[0] - 1
    u_f = u[:, :, :nx] if u.shape[-1] == nx + 1 else u  # (nz, ny, nx) mass-x faces

    # rho at mass levels (nz,ny,nx); rho at w faces via fzm/fzp -> simple average
    # (uniform dz neutral slab): rho_f(k) = 0.5*(rho(k)+rho(k-1)), faces 1..nz-1.
    rho_m = rho  # (nz, ny, nx)
    rho_f = jnp.zeros((nz + 1,) + tuple(rho.shape[1:]), dtype=rho.dtype)
    rho_f = rho_f.at[1:nz, :, :].set(0.5 * (rho_m[1:nz, :, :] + rho_m[0 : nz - 1, :, :]))

    # ---- D11 = 2 du/dx at mass-x (u faces) -> titau11 = -rho*K*D11 at mass cells ----
    # du/dx centered to mass cell i: (u(i+1)-u(i))/dx with u on x faces; here u_f(i)
    # is the left x-face of mass cell i, so du/dx at cell i = (u_f(i+1)-u_f(i))/dx.
    dudx = (jnp.roll(u_f, -1, axis=2) - u_f) / dx  # (nz,ny,nx) at mass cells
    titau11 = -rho_m * k * (2.0 * dudx)  # at mass cells i
    # d/dx(titau11) back to u faces: (titau11(i)-titau11(i-1))/dx.
    du = (titau11 - jnp.roll(titau11, 1, axis=2)) / dx  # (nz,ny,nx) at u faces

    # ---- D33 = 2 dw/dz at mass cells -> titau33 = -rho*K*D33 ----
    dwdz = (w[1 : nz + 1, :, :] - w[0:nz, :, :]) / dz  # (nz,ny,nx) at mass cells
    titau33 = -rho_m * k * (2.0 * dwdz)
    # d/dz(titau33) to w faces 1..nz-1: (titau33(k)-titau33(k-1))/dz.
    dw = jnp.zeros_like(w)
    if nz >= 2:
        dw_int = (titau33[1:nz, :, :] - titau33[0 : nz - 1, :, :]) / dz
        dw = dw.at[1:nz, :, :].set(dw_int)

    # ---- D13 = dw/dx + du/dz at w faces (vorticity-point stagger) ----
    # dw/dx at w faces (mass-x): centered (w(i+1)-w(i-1))/(2dx).
    dwdx_f = _ddx_periodic(w, dx, axis=2)  # (nz+1,ny,nx) at w faces
    # du/dz at w faces 1..nz-1: (u_f(k)-u_f(k-1))/dz; faces 0,nz -> 0 (rigid).
    dudz_f = jnp.zeros_like(w)
    if nz >= 2:
        dudz_f = dudz_f.at[1:nz, :, :].set((u_f[1:nz, :, :] - u_f[0 : nz - 1, :, :]) / dz)
    D13 = dwdx_f + dudz_f  # (nz+1,ny,nx) at w faces
    titau13 = -rho_f * k * D13  # rho at w faces; faces 0,nz carry rho_f=0 -> 0 stress
    # u gets d/dz(titau13) at mass levels (u stagger): (titau13(k+1)-titau13(k))/dz.
    du = du + (titau13[1 : nz + 1, :, :] - titau13[0:nz, :, :]) / dz
    # w gets d/dx(titau13) at w faces: centered (titau13(i+1)-titau13(i-1))/(2dx).
    dw = dw + _ddx_periodic(titau13, dx, axis=2)

    # NOTE: titau is -rho*K*D, so d/dx(titau11) etc. already carry the minus sign;
    # the WRF tendency is +g*dz/dnw*rdx*(titau11(i)-titau11(i-1)) and with the
    # WRF-signed dnw<0 the net diffusive sign is +K d2u/dx2 (down-gradient).  We
    # have folded the column factor into rho weighting; the operator above is the
    # density-weighted stress divergence d/dx_j(rho*K*D_ij) which is +rho*K*Lap to
    # leading order (down-gradient).  Flip the assembled sign so du/dt is down-gradient.
    du = -du
    dw = -dw

    if u.shape[-1] == nx + 1:
        du = jnp.concatenate((du, du[:, :, :1]), axis=2)
    return du, dw


def deformation_components_3d(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    *,
    dx_m: float,
    dy_m: float,
    msftx: jax.Array | None = None,
    msfty: jax.Array | None = None,
    msfux: jax.Array | None = None,
    msfuy: jax.Array | None = None,
    msfvx: jax.Array | None = None,
    msfvy: jax.Array | None = None,
    zx: jax.Array | None = None,
    zy: jax.Array | None = None,
    rdz: jax.Array | None = None,
    rdzw: jax.Array | None = None,
    fnm: jax.Array | None = None,
    fnp: jax.Array | None = None,
    cf1: jax.Array | float | None = None,
    cf2: jax.Array | float | None = None,
    cf3: jax.Array | float | None = None,
    dn: jax.Array | None = None,
    dnw: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """WRF ``cal_deform_and_div`` D11/D22/D33/D12/D13/D23 components.

    This is the periodic-x/y real-terrain form used by WRF diffusion over
    orography.  It intentionally models the interior periodic stencil only; open,
    specified, and nested boundary degradation remains outside this helper.
    """

    d11, d22, d12 = horizontal_deformation_2d(
        u,
        v,
        dx_m=dx_m,
        dy_m=dy_m,
        msftx=msftx,
        msfty=msfty,
        msfux=msfux,
        msfuy=msfuy,
        msfvx=msfvx,
        msfvy=msfvy,
        zx=zx,
        zy=zy,
        rdzw=rdzw,
        fnm=fnm,
        fnp=fnp,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        dn=dn,
        dnw=dnw,
    )

    nx = w.shape[-1]
    ny = w.shape[1]
    nz = w.shape[0] - 1
    dx = float(dx_m)
    dy = float(dy_m)
    u_m = _core_x(u, nx)
    v_m = _core_y(v, ny)
    msftx_m = _mass_metric_or_one(msftx, u_m)
    msfty_m = _mass_metric_or_one(msfty, u_m)
    msfux_u = _x_metric_or_one(msfux, nx=nx, reference=u_m)
    msfuy_u = _x_metric_or_one(msfuy, nx=nx, reference=u_m)
    msfvx_v = _y_metric_or_one(msfvx, ny=ny, reference=v_m)
    msfvy_v = _y_metric_or_one(msfvy, ny=ny, reference=v_m)
    zx_f = _zface_or_zero(zx, nz=nz, ny=ny, nx=nx, reference=u_m)
    zy_f = _zface_or_zero(zy, nz=nz, ny=ny, nx=nx, reference=u_m)
    rdz_f = _wface3_or_one(rdz, nz=nz, ny=ny, nx=nx, reference=w)
    rdzw_m = _mass3_or_one(rdzw, u_m)

    d33 = 2.0 * (w[1 : nz + 1, :, :] - w[0:nz, :, :]) * rdzw_m

    # D13 = m_u^2*(d(w/msfty)/dx - zx*d(w/msfty)/dpsi) + du/dz.
    w_yhat = w / msfty_m[None, :, :]
    w_yhat_w = jnp.roll(w_yhat, 1, axis=2)
    w_yhat_avg_m = 0.25 * (
        w_yhat[0:nz] + w_yhat[1 : nz + 1] + w_yhat_w[0:nz] + w_yhat_w[1 : nz + 1]
    )
    rdz_u = 0.5 * (rdz_f[1:nz, :, :] + jnp.roll(rdz_f[1:nz, :, :], 1, axis=2))
    d13_int = (
        (msfux_u * msfuy_u)[None, :, :]
        * (
            (w_yhat[1:nz, :, :] - w_yhat_w[1:nz, :, :]) / dx
            - (w_yhat_avg_m[1:nz, :, :] - w_yhat_avg_m[0 : nz - 1, :, :])
            * zx_f[1:nz, :, :]
            * rdz_u
        )
        + (u_m[1:nz, :, :] - u_m[0 : nz - 1, :, :]) * rdz_u
    )
    d13 = jnp.zeros_like(w)
    if nz >= 2:
        d13 = d13.at[1:nz, :, :].set(d13_int)

    # D23 = m_v^2*(d(w/msftx)/dy - zy*d(w/msftx)/dpsi) + dv/dz.
    w_xhat = w / msftx_m[None, :, :]
    w_xhat_s = jnp.roll(w_xhat, 1, axis=1)
    w_xhat_avg_m = 0.25 * (
        w_xhat[0:nz] + w_xhat[1 : nz + 1] + w_xhat_s[0:nz] + w_xhat_s[1 : nz + 1]
    )
    rdz_v = 0.5 * (rdz_f[1:nz, :, :] + jnp.roll(rdz_f[1:nz, :, :], 1, axis=1))
    d23_int = (
        (msfvx_v * msfvy_v)[None, :, :]
        * (
            (w_xhat[1:nz, :, :] - w_xhat_s[1:nz, :, :]) / dy
            - (w_xhat_avg_m[1:nz, :, :] - w_xhat_avg_m[0 : nz - 1, :, :])
            * zy_f[1:nz, :, :]
            * rdz_v
        )
        + (v_m[1:nz, :, :] - v_m[0 : nz - 1, :, :]) * rdz_v
    )
    d23 = jnp.zeros_like(w)
    if nz >= 2:
        d23 = d23.at[1:nz, :, :].set(d23_int)

    return d11, d22, d33, d12, d13, d23


def wrf_terrain_deformation_momentum_tendency(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    *,
    rho: jax.Array,
    xkmh: jax.Array | float,
    xkmv: jax.Array | float | None = None,
    dx_m: float,
    dy_m: float,
    msftx: jax.Array | None = None,
    msfty: jax.Array | None = None,
    msfux: jax.Array | None = None,
    msfuy: jax.Array | None = None,
    msfvx: jax.Array | None = None,
    msfvy: jax.Array | None = None,
    zx: jax.Array | None = None,
    zy: jax.Array | None = None,
    rdz: jax.Array | None = None,
    rdzw: jax.Array | None = None,
    dn: jax.Array | None = None,
    dnw: jax.Array | None = None,
    fnm: jax.Array | None = None,
    fnp: jax.Array | None = None,
    cf1: jax.Array | float | None = None,
    cf2: jax.Array | float | None = None,
    cf3: jax.Array | float | None = None,
    gravity: float = GRAVITY_M_S2,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """WRF terrain-following deformation-tensor momentum diffusion.

    Implements the periodic interior of WRF ``horizontal_diffusion_{u,v,w}_2`` and
    ``vertical_diffusion_{u,v,w}_2`` for real map factors and nonzero
    ``zx``/``zy`` cross-coordinate metric terms.  The returned tendencies are
    WRF's mass/density-coupled ``ru_tendf``, ``rv_tendf``, and ``rw_tendf`` on the
    native u/v/w staggers.  Constant-K diffusion passes scalar ``xkmh``/``xkmv``;
    Smagorinsky/deformation paths can pass spatially varying arrays.
    """

    nx = w.shape[-1]
    ny = w.shape[1]
    nz = w.shape[0] - 1
    dx = float(dx_m)
    dy = float(dy_m)
    dtype = rho.dtype
    u_m = _core_x(u, nx)
    v_m = _core_y(v, ny)
    rho_m = jnp.asarray(rho, dtype=dtype)
    if jnp.ndim(jnp.asarray(xkmh)) == 0:
        xkmh_m = jnp.full_like(rho_m, jnp.asarray(xkmh, dtype=dtype))
    else:
        xkmh_m = jnp.asarray(xkmh, dtype=dtype)
    if xkmv is None:
        xkmv_m = xkmh_m
    elif jnp.ndim(jnp.asarray(xkmv)) == 0:
        xkmv_m = jnp.full_like(rho_m, jnp.asarray(xkmv, dtype=dtype))
    else:
        xkmv_m = jnp.asarray(xkmv, dtype=dtype)

    msftx_m = _mass_metric_or_one(msftx, rho_m)
    msfty_m = _mass_metric_or_one(msfty, rho_m)
    msfux_u = _x_metric_or_one(msfux, nx=nx, reference=u_m)
    msfuy_u = _x_metric_or_one(msfuy, nx=nx, reference=u_m)
    msfvx_v = _y_metric_or_one(msfvx, ny=ny, reference=v_m)
    msfvy_v = _y_metric_or_one(msfvy, ny=ny, reference=v_m)
    zx_f = _zface_or_zero(zx, nz=nz, ny=ny, nx=nx, reference=rho_m)
    zy_f = _zface_or_zero(zy, nz=nz, ny=ny, nx=nx, reference=rho_m)
    rdz_f = _wface3_or_one(rdz, nz=nz, ny=ny, nx=nx, reference=w)
    rdzw_m = _mass3_or_one(rdzw, rho_m)
    if dn is None:
        dn_v = -jnp.ones((nz,), dtype=dtype)
    else:
        dn_v = jnp.asarray(dn, dtype=dtype)
    if dnw is None:
        dnw_v = -jnp.ones((nz,), dtype=dtype)
    else:
        dnw_v = jnp.asarray(dnw, dtype=dtype)
    if fnm is None:
        fnm_v = jnp.ones((nz,), dtype=dtype)
        fnp_v = jnp.zeros((nz,), dtype=dtype)
    else:
        fnm_v = jnp.asarray(fnm, dtype=dtype)
        fnp_v = jnp.asarray(fnp, dtype=dtype) if fnp is not None else jnp.zeros((nz,), dtype=dtype)

    d11, d22, d33, d12, d13, d23 = deformation_components_3d(
        u,
        v,
        w,
        dx_m=dx,
        dy_m=dy,
        msftx=msftx,
        msfty=msfty,
        msfux=msfux,
        msfuy=msfuy,
        msfvx=msfvx,
        msfvy=msfvy,
        zx=zx,
        zy=zy,
        rdz=rdz,
        rdzw=rdzw,
        fnm=fnm,
        fnp=fnp,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        dn=dn,
        dnw=dnw,
    )

    tau11 = -rho_m * xkmh_m * d11
    tau22 = -rho_m * xkmh_m * d22
    tau33 = -rho_m * xkmh_m * d33

    rho_c = 0.25 * (
        rho_m
        + jnp.roll(rho_m, 1, axis=2)
        + jnp.roll(rho_m, 1, axis=1)
        + jnp.roll(jnp.roll(rho_m, 1, axis=1), 1, axis=2)
    )
    k_c = 0.25 * (
        xkmh_m
        + jnp.roll(xkmh_m, 1, axis=2)
        + jnp.roll(xkmh_m, 1, axis=1)
        + jnp.roll(jnp.roll(xkmh_m, 1, axis=1), 1, axis=2)
    )
    tau12 = -rho_c * k_c * d12

    tau13 = jnp.zeros_like(w)
    tau23 = jnp.zeros_like(w)
    if nz >= 2:
        rho_w = jnp.roll(rho_m, 1, axis=2)
        k_w = jnp.roll(xkmv_m, 1, axis=2)
        rho13 = 0.5 * (
            fnm_v[1:nz, None, None] * (rho_m[1:nz] + rho_w[1:nz])
            + fnp_v[1:nz, None, None] * (rho_m[0 : nz - 1] + rho_w[0 : nz - 1])
        )
        k13 = 0.5 * (
            fnm_v[1:nz, None, None] * (xkmv_m[1:nz] + k_w[1:nz])
            + fnp_v[1:nz, None, None] * (xkmv_m[0 : nz - 1] + k_w[0 : nz - 1])
        )
        tau13 = tau13.at[1:nz, :, :].set(-rho13 * k13 * d13[1:nz, :, :])

        rho_s = jnp.roll(rho_m, 1, axis=1)
        k_s = jnp.roll(xkmv_m, 1, axis=1)
        rho23 = 0.5 * (
            fnm_v[1:nz, None, None] * (rho_m[1:nz] + rho_s[1:nz])
            + fnp_v[1:nz, None, None] * (rho_m[0 : nz - 1] + rho_s[0 : nz - 1])
        )
        k23 = 0.5 * (
            fnm_v[1:nz, None, None] * (xkmv_m[1:nz] + k_s[1:nz])
            + fnp_v[1:nz, None, None] * (xkmv_m[0 : nz - 1] + k_s[0 : nz - 1])
        )
        tau23 = tau23.at[1:nz, :, :].set(-rho23 * k23 * d23[1:nz, :, :])

    # u horizontal stress divergence plus terrain-slope metric terms.
    tau11avg = jnp.zeros((nz + 1, ny, nx), dtype=dtype)
    tau12avg = jnp.zeros((nz + 1, ny, nx), dtype=dtype)
    if nz >= 2:
        tau11avg = tau11avg.at[1:nz, :, :].set(
            0.5
            * (
                fnm_v[1:nz, None, None] * (tau11[1:nz] + jnp.roll(tau11[1:nz], 1, axis=2))
                + fnp_v[1:nz, None, None]
                * (tau11[0 : nz - 1] + jnp.roll(tau11[0 : nz - 1], 1, axis=2))
            )
        )
        tau12avg = tau12avg.at[1:nz, :, :].set(
            0.5
            * (
                fnm_v[1:nz, None, None] * (jnp.roll(tau12[1:nz], -1, axis=1) + tau12[1:nz])
                + fnp_v[1:nz, None, None]
                * (jnp.roll(tau12[0 : nz - 1], -1, axis=1) + tau12[0 : nz - 1])
            )
        )
    tmpdz_u = 0.5 * (1.0 / rdzw_m + 1.0 / jnp.roll(rdzw_m, 1, axis=2))
    zx_at_u = 0.5 * (zx_f[:-1] + zx_f[1:])
    zy_at_u = 0.125 * (
        jnp.roll(zy_f[:-1], 1, axis=2)
        + zy_f[:-1]
        + jnp.roll(jnp.roll(zy_f[:-1], 1, axis=2), -1, axis=1)
        + jnp.roll(zy_f[:-1], -1, axis=1)
        + jnp.roll(zy_f[1:], 1, axis=2)
        + zy_f[1:]
        + jnp.roll(jnp.roll(zy_f[1:], 1, axis=2), -1, axis=1)
        + jnp.roll(zy_f[1:], -1, axis=1)
    )
    du = (
        float(gravity)
        * tmpdz_u
        / dnw_v[:, None, None]
        * (
            msfux_u[None, :, :] * (1.0 / dx) * (tau11 - jnp.roll(tau11, 1, axis=2))
            + msfuy_u[None, :, :] * (1.0 / dy) * (jnp.roll(tau12, -1, axis=1) - tau12)
            - msfux_u[None, :, :] * zx_at_u * (tau11avg[1:] - tau11avg[:-1]) / tmpdz_u
            - msfuy_u[None, :, :] * zy_at_u * (tau12avg[1:] - tau12avg[:-1]) / tmpdz_u
        )
    )
    if nz >= 2:
        du = du.at[1:nz, :, :].add(
            float(gravity) / dnw_v[1:nz, None, None] * (tau13[2 : nz + 1] - tau13[1:nz])
        )
    du = du.at[0, :, :].add(float(gravity) / dnw_v[0] * tau13[1, :, :])

    # v horizontal + vertical stress divergence.
    tau21avg = jnp.zeros((nz + 1, ny, nx), dtype=dtype)
    tau22avg = jnp.zeros((nz + 1, ny, nx), dtype=dtype)
    if nz >= 2:
        tau21avg = tau21avg.at[1:nz, :, :].set(
            0.5
            * (
                fnm_v[1:nz, None, None] * (jnp.roll(tau12[1:nz], -1, axis=2) + tau12[1:nz])
                + fnp_v[1:nz, None, None]
                * (jnp.roll(tau12[0 : nz - 1], -1, axis=2) + tau12[0 : nz - 1])
            )
        )
        tau22avg = tau22avg.at[1:nz, :, :].set(
            0.5
            * (
                fnm_v[1:nz, None, None] * (jnp.roll(tau22[1:nz], 1, axis=1) + tau22[1:nz])
                + fnp_v[1:nz, None, None]
                * (jnp.roll(tau22[0 : nz - 1], 1, axis=1) + tau22[0 : nz - 1])
            )
        )
    tmpdz_v = 0.5 * (1.0 / rdzw_m + 1.0 / jnp.roll(rdzw_m, 1, axis=1))
    zx_at_v = 0.125 * (
        zx_f[:-1]
        + jnp.roll(zx_f[:-1], -1, axis=2)
        + jnp.roll(zx_f[:-1], 1, axis=1)
        + jnp.roll(jnp.roll(zx_f[:-1], -1, axis=2), 1, axis=1)
        + zx_f[1:]
        + jnp.roll(zx_f[1:], -1, axis=2)
        + jnp.roll(zx_f[1:], 1, axis=1)
        + jnp.roll(jnp.roll(zx_f[1:], -1, axis=2), 1, axis=1)
    )
    zy_at_v = 0.5 * (zy_f[:-1] + zy_f[1:])
    dv = (
        float(gravity)
        * tmpdz_v
        / dnw_v[:, None, None]
        * (
            msfvy_v[None, :, :] * (1.0 / dy) * (tau22 - jnp.roll(tau22, 1, axis=1))
            + msfvx_v[None, :, :] * (1.0 / dx) * (jnp.roll(tau12, -1, axis=2) - tau12)
            - msfvx_v[None, :, :] * zx_at_v * (tau21avg[1:] - tau21avg[:-1]) / tmpdz_v
            - msfvy_v[None, :, :] * zy_at_v * (tau22avg[1:] - tau22avg[:-1]) / tmpdz_v
        )
    )
    if nz >= 2:
        dv = dv.at[1:nz, :, :].add(
            float(gravity) / dnw_v[1:nz, None, None] * (tau23[2 : nz + 1] - tau23[1:nz])
        )
    dv = dv.at[0, :, :].add(float(gravity) / dnw_v[0] * tau23[1, :, :])

    # w horizontal + vertical stress divergence on interior w faces.
    dw = jnp.zeros_like(w)
    if nz >= 2:
        tau13avg_m = 0.25 * (
            jnp.roll(tau13[1 : nz + 1], -1, axis=2)
            + tau13[1 : nz + 1]
            + jnp.roll(tau13[0:nz], -1, axis=2)
            + tau13[0:nz]
        )
        tau23avg_m = 0.25 * (
            jnp.roll(tau23[1 : nz + 1], -1, axis=1)
            + tau23[1 : nz + 1]
            + jnp.roll(tau23[0:nz], -1, axis=1)
            + tau23[0:nz]
        )
        zx_at_w = 0.5 * (zx_f[1:nz] + jnp.roll(zx_f[1:nz], -1, axis=2))
        zy_at_w = 0.5 * (zy_f[1:nz] + jnp.roll(zy_f[1:nz], -1, axis=1))
        dw_h = (
            float(gravity)
            / (dn_v[1:nz, None, None] * rdz_f[1:nz])
            * (
                msftx_m[None, :, :] * (1.0 / dx) * (jnp.roll(tau13[1:nz], -1, axis=2) - tau13[1:nz])
                + msfty_m[None, :, :] * (1.0 / dy) * (jnp.roll(tau23[1:nz], -1, axis=1) - tau23[1:nz])
                - msfty_m[None, :, :]
                * rdz_f[1:nz]
                * (
                    zx_at_w * (tau13avg_m[1:nz] - tau13avg_m[0 : nz - 1])
                    + zy_at_w * (tau23avg_m[1:nz] - tau23avg_m[0 : nz - 1])
                )
            )
        )
        dw_v = float(gravity) * (tau33[1:nz] - tau33[0 : nz - 1]) / dn_v[1:nz, None, None]
        dw = dw.at[1:nz, :, :].set(dw_h + dw_v)

    if u.shape[-1] == nx + 1:
        du = jnp.concatenate((du, du[:, :, :1]), axis=2)
    if v.shape[1] == ny + 1:
        dv = jnp.concatenate((dv, dv[:, :1, :]), axis=1)
    return du, dv, dw


# Smagorinsky / Prandtl constants (WRF share/module_model_constants.F:86 and the
# Registry default for c_s).  prandtl = 1/3 so the heat eddy diffusivity xkhh is
# 3x the momentum xkmh (= the WRF khdq=3*khdif convention for the perturbation
# scalar in horizontal_diffusion_3dmp).
PRANDTL = 1.0 / 3.0
C_S_DEFAULT = 0.25


def horizontal_deformation_2d(
    u: jax.Array,
    v: jax.Array,
    *,
    dx_m: float,
    dy_m: float,
    msftx: jax.Array | None = None,
    msfty: jax.Array | None = None,
    msfux: jax.Array | None = None,
    msfuy: jax.Array | None = None,
    msfvx: jax.Array | None = None,
    msfvy: jax.Array | None = None,
    zx: jax.Array | None = None,
    zy: jax.Array | None = None,
    rdzw: jax.Array | None = None,
    fnm: jax.Array | None = None,
    fnp: jax.Array | None = None,
    cf1: jax.Array | float | None = None,
    cf2: jax.Array | float | None = None,
    cf3: jax.Array | float | None = None,
    dn: jax.Array | None = None,
    dnw: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """WRF ``cal_deform_and_div`` horizontal deformations.

    Returns ``(D11, D22, D12)`` on mass/corner points.  With all optional metric
    inputs omitted this is the historical flat periodic slab reduction.  When
    map factors and ``zx``/``zy`` are supplied it follows WRF
    ``module_diffusion_em.F:139-681``:

      * ``D11 = 2*m^2*(d(u/msfuy)/dx - zx*d(u/msfuy)/dpsi)`` at mass points.
      * ``D22 = 2*m^2*(d(v/msfvx)/dy - zy*d(v/msfvx)/dpsi)`` at mass points.
      * ``D12 = m^2*(d(u/msfux)/dy - zy*d(u/msfux)/dpsi
                   + d(v/msfvy)/dx - zx*d(v/msfvy)/dpsi)`` at vorticity points.

    ``zx``/``zy`` are WRF z-face slope metrics with shape ``(nz+1, ny, nx)``;
    ``rdzw`` is the mass-level inverse vertical spacing ``(nz, ny, nx)``.  The
    ``fnm/fnp/cf*/dn/dnw`` arguments are WRF vertical interpolation/extrapolation
    coefficients.  Flat defaults are ``msf=1`` and ``zx=zy=0``.

    Source: ``module_diffusion_em.F:cal_deform_and_div`` (Chen & Dudhia 2000
    eqns 13a/13b/13d):

      * ``D11 = 2 m^2 (du^/dX + zx du^/dpsi)`` -> ``2 du/dx`` at mass points
        (``:177-218``).
      * ``D22 = 2 m^2 (dv^/dY + zy dv^/dpsi)`` -> ``2 dv/dy`` at mass points
        (``:287-328``).
      * ``D12 = m^2 (dv^/dX + du^/dY + zx dv^/dpsi + zy du^/dpsi)`` ->
        ``du/dy + dv/dx`` at vorticity (cell-corner) points (``:508-627``).

    Staggering (periodic x and y):
      * ``u`` on x-faces ``(nz, ny, nx+1)`` -- face ``i`` is the WEST face of mass
        cell ``i`` (face ``nx`` == face ``0``).
      * ``v`` on y-faces ``(nz, ny+1, nx)`` -- face ``j`` is the SOUTH face of mass
        cell ``j`` (face ``ny`` == face ``0``).
      * ``D11``/``D22`` on mass cells ``(nz, ny, nx)``.
      * ``D12`` on cell corners ``(nz, ny, nx)`` indexed so corner ``(i,j)`` is the
        SW corner of mass cell ``(i,j)`` (WRF vorticity-point convention: it uses
        ``u(i,j)-u(i,j-1)`` and ``v(i,j)-v(i-1,j)``).
    """

    dx = float(dx_m)
    dy = float(dy_m)
    nx = u.shape[-1] - 1 if u.shape[-1] > v.shape[-1] else u.shape[-1]
    ny = v.shape[1] - 1 if v.shape[1] > u.shape[1] else v.shape[1]
    nz = u.shape[0]

    u_m = _core_x(u, nx)
    v_m = _core_y(v, ny)
    msftx_m = _mass_metric_or_one(msftx, u_m)
    msfty_m = _mass_metric_or_one(msfty, u_m)
    msfux_u = _x_metric_or_one(msfux, nx=nx, reference=u_m)
    msfuy_u = _x_metric_or_one(msfuy, nx=nx, reference=u_m)
    msfvx_v = _y_metric_or_one(msfvx, ny=ny, reference=v_m)
    msfvy_v = _y_metric_or_one(msfvy, ny=ny, reference=v_m)
    zx_f = _zface_or_zero(zx, nz=nz, ny=ny, nx=nx, reference=u_m)
    zy_f = _zface_or_zero(zy, nz=nz, ny=ny, nx=nx, reference=u_m)
    rdzw_m = _mass3_or_one(rdzw, u_m)

    # D11 = 2*m^2*(d(u/msfuy)/dx - zx*d(u/msfuy)/dpsi).  WRF averages the
    # transformed u to z-faces before the slope derivative.
    u_yhat = u_m / msfuy_u[None, :, :]
    u_yhat_e = jnp.roll(u_yhat, -1, axis=2)
    u_yhat_zf = _vertical_face_average_pair(
        u_yhat, u_yhat_e, fnm=fnm, fnp=fnp, cf1=cf1, cf2=cf2, cf3=cf3, dn=dn, dnw=dnw
    )
    zx_d11 = 0.25 * (zx_f[:-1] + jnp.roll(zx_f[:-1], -1, axis=2) + zx_f[1:] + jnp.roll(zx_f[1:], -1, axis=2))
    d11_core = msftx_m[None, :, :] * msfty_m[None, :, :] * (
        (u_yhat_e - u_yhat) / dx
        - (u_yhat_zf[1:] - u_yhat_zf[:-1]) * zx_d11 * rdzw_m
    )
    d11 = 2.0 * d11_core

    # D22 = 2*m^2*(d(v/msfvx)/dy - zy*d(v/msfvx)/dpsi).
    v_xhat = v_m / msfvx_v[None, :, :]
    v_xhat_n = jnp.roll(v_xhat, -1, axis=1)
    v_xhat_zf = _vertical_face_average_pair(
        v_xhat, v_xhat_n, fnm=fnm, fnp=fnp, cf1=cf1, cf2=cf2, cf3=cf3, dn=dn, dnw=dnw
    )
    zy_d22 = 0.25 * (zy_f[:-1] + jnp.roll(zy_f[:-1], -1, axis=1) + zy_f[1:] + jnp.roll(zy_f[1:], -1, axis=1))
    d22_core = msftx_m[None, :, :] * msfty_m[None, :, :] * (
        (v_xhat_n - v_xhat) / dy
        - (v_xhat_zf[1:] - v_xhat_zf[:-1]) * zy_d22 * rdzw_m
    )
    d22 = 2.0 * d22_core

    # D12 corner metric: 0.25*(msfux(i,j-1)+msfux(i,j))*(msfvy(i-1,j)+msfvy(i,j)).
    mm_corner = 0.25 * (
        jnp.roll(msfux_u, 1, axis=0) + msfux_u
    ) * (
        jnp.roll(msfvy_v, 1, axis=1) + msfvy_v
    )
    rdzw_corner = 0.25 * (
        rdzw_m
        + jnp.roll(rdzw_m, 1, axis=1)
        + jnp.roll(rdzw_m, 1, axis=2)
        + jnp.roll(jnp.roll(rdzw_m, 1, axis=1), 1, axis=2)
    )

    # First D12 half: d(u/msfux)/dy - zy*d(u/msfux)/dpsi at vorticity points.
    u_xhat = u_m / msfux_u[None, :, :]
    u_xhat_s = jnp.roll(u_xhat, 1, axis=1)
    u_xhat_zf = _vertical_face_average_pair(
        u_xhat_s, u_xhat, fnm=fnm, fnp=fnp, cf1=cf1, cf2=cf2, cf3=cf3, dn=dn, dnw=dnw
    )
    zy_d12 = 0.25 * (
        jnp.roll(zy_f[:-1], 1, axis=2)
        + zy_f[:-1]
        + jnp.roll(zy_f[1:], 1, axis=2)
        + zy_f[1:]
    )
    dudy = (u_xhat - u_xhat_s) / dy
    d12_u = mm_corner[None, :, :] * (
        dudy - (u_xhat_zf[1:] - u_xhat_zf[:-1]) * zy_d12 * rdzw_corner
    )

    # Second D12 half: d(v/msfvy)/dx - zx*d(v/msfvy)/dpsi.
    v_yhat = v_m / msfvy_v[None, :, :]
    v_yhat_w = jnp.roll(v_yhat, 1, axis=2)
    v_yhat_zf = _vertical_face_average_pair(
        v_yhat_w, v_yhat, fnm=fnm, fnp=fnp, cf1=cf1, cf2=cf2, cf3=cf3, dn=dn, dnw=dnw
    )
    zx_d12 = 0.25 * (
        jnp.roll(zx_f[:-1], 1, axis=1)
        + zx_f[:-1]
        + jnp.roll(zx_f[1:], 1, axis=1)
        + zx_f[1:]
    )
    dvdx = (v_yhat - v_yhat_w) / dx
    d12_v = mm_corner[None, :, :] * (
        dvdx - (v_yhat_zf[1:] - v_yhat_zf[:-1]) * zx_d12 * rdzw_corner
    )
    d12 = d12_u + d12_v
    return d11, d22, d12


def smag2d_horizontal_km(
    d11: jax.Array,
    d22: jax.Array,
    d12: jax.Array,
    *,
    dx_m: float,
    dy_m: float,
    c_s: float = C_S_DEFAULT,
    prandtl: float = PRANDTL,
    msftx: jax.Array | None = None,
    msfty: jax.Array | None = None,
    zx: jax.Array | None = None,
    zy: jax.Array | None = None,
    rdzw: jax.Array | None = None,
    diff_opt_for_slope: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """WRF ``smag2d_km`` 2-D Smagorinsky horizontal eddy viscosity (km_opt=4).

    Returns ``(xkmh, xkhh)`` -- the horizontal MOMENTUM and HEAT eddy
    diffusivities (m^2/s) on mass cells.  Faithful transcription of
    ``module_diffusion_em.F:smag2d_km:2001-2042``:

      * ``def2 = 0.25*(D11-D22)^2 + tmp^2`` with
        ``tmp = 0.25*(D12(i,j)+D12(i,j+1)+D12(i+1,j)+D12(i+1,j+1))`` -- the four
        surrounding corner ``D12`` values averaged onto the mass cell (``:2004-2007``).
      * ``mlen_h = sqrt(dx/msftx * dy/msfty)`` (``:2015``).
      * ``xkmh = c_s^2 * mlen_h^2 * sqrt(def2)`` then capped ``min(xkmh, 10*mlen_h)``
        (``:2018-2019``).  ``c_s`` default 0.25 (Registry), ``prandtl`` = 1/3.
      * ``xkhh = xkmh / prandtl`` (``:2021``).

    When ``diff_opt_for_slope == 2`` and ``zx``/``zy``/``rdzw`` are supplied, WRF's
    terrain-slope reduction (``:2023-2039``) is applied.  The operational
    ``diff_opt=1`` Smagorinsky path should leave ``diff_opt_for_slope`` at the
    default 1, exactly as WRF does.

    D12 is on cell corners with corner ``(i,j)`` == SW corner of cell ``(i,j)``
    (see :func:`horizontal_deformation_2d`).  WRF's four-corner average of cell
    ``(i,j)`` uses corners ``(i,j),(i,j+1),(i+1,j),(i+1,j+1)`` = SW, NW, SE, NE of
    the cell, reproduced here with periodic rolls.
    """

    dx = float(dx_m)
    dy = float(dy_m)
    cs = float(c_s)
    pr = float(prandtl)

    # four-corner average of D12 onto mass cell (i,j): SW + NW(j+1) + SE(i+1) + NE.
    d12_nw = jnp.roll(d12, -1, axis=1)  # corner (i, j+1)
    d12_se = jnp.roll(d12, -1, axis=2)  # corner (i+1, j)
    d12_ne = jnp.roll(jnp.roll(d12, -1, axis=1), -1, axis=2)  # corner (i+1, j+1)
    tmp = 0.25 * (d12 + d12_nw + d12_se + d12_ne)

    def2 = 0.25 * (d11 - d22) * (d11 - d22) + tmp * tmp
    msftx_m = _mass_metric_or_one(msftx, d11)
    msfty_m = _mass_metric_or_one(msfty, d11)
    mlen_h = jnp.sqrt((dx / msftx_m[None, :, :]) * (dy / msfty_m[None, :, :]))
    xkmh = cs * cs * mlen_h * mlen_h * jnp.sqrt(def2)
    # WRF :2019 cap (NOT a tuning clamp -- it is the literal smag2d_km ceiling that
    # bounds K so the explicit horizontal diffusion stays inside its stability
    # envelope; faithful transcription).
    xkmh = jnp.minimum(xkmh, 10.0 * mlen_h)

    if int(diff_opt_for_slope) == 2 and zx is not None and zy is not None and rdzw is not None:
        nz, ny, nx = d11.shape
        zx_f = _zface_or_zero(zx, nz=nz, ny=ny, nx=nx, reference=d11)
        zy_f = _zface_or_zero(zy, nz=nz, ny=ny, nx=nx, reference=d11)
        rdzw_m = _mass3_or_one(rdzw, d11)
        dxm = dx / msftx_m[None, :, :]
        dym = dy / msfty_m[None, :, :]
        tmpzx = (
            0.25
            * (
                jnp.abs(zx_f[:-1])
                + jnp.abs(jnp.roll(zx_f[:-1], -1, axis=2))
                + jnp.abs(zx_f[1:])
                + jnp.abs(jnp.roll(zx_f[1:], -1, axis=2))
            )
            * rdzw_m
            * dxm
        )
        tmpzy = (
            0.25
            * (
                jnp.abs(zy_f[:-1])
                + jnp.abs(jnp.roll(zy_f[:-1], -1, axis=1))
                + jnp.abs(zy_f[1:])
                + jnp.abs(jnp.roll(zy_f[1:], -1, axis=1))
            )
            * rdzw_m
            * dym
        )
        alpha = jnp.maximum(jnp.sqrt(tmpzx * tmpzx + tmpzy * tmpzy), 1.0)
        def_limit = jnp.maximum(10.0 / mlen_h, 1.0e-3)
        slope_divisor = jnp.where(jnp.sqrt(def2) > def_limit, alpha * alpha, alpha)
        xkmh = xkmh / slope_divisor
    xkhh = xkmh / pr
    return xkmh, xkhh


def _xkmhd_to_w_levels(xkmhd_h: jax.Array, nz_faces: int) -> jax.Array:
    """Average the mass-cell horizontal viscosity to the w (z-face) levels.

    WRF (``horizontal_diffusion`` 'w' branch :2887) averages xkmhd in z to the w
    level: ``0.5*(xkmhd(k)+xkmhd(k-1))``.  ``w`` has ``nz_faces = nz+1`` levels
    (faces); interior faces 1..nz-1 use the adjacent-mass-level average, and faces
    0 / nz take the nearest interior level (rigid; bounding-face w diffusion is moot
    under the rigid lid / surface).
    """

    nz = xkmhd_h.shape[0]
    face = jnp.zeros((nz_faces,) + tuple(xkmhd_h.shape[1:]), dtype=xkmhd_h.dtype)
    face = face.at[1:nz, :, :].set(0.5 * (xkmhd_h[1:nz, :, :] + xkmhd_h[0 : nz - 1, :, :]))
    face = face.at[0, :, :].set(xkmhd_h[0, :, :])
    face = face.at[nz, :, :].set(xkmhd_h[nz - 1, :, :])
    return face


def _hdiff_coord_scalar(
    field: jax.Array,
    xkmhd: jax.Array,
    mass: jax.Array,
    *,
    dx_m: float,
    dy_m: float,
) -> jax.Array:
    """diff_opt=1 coordinate-surface variable-K flux divergence (scalar branch).

    Faithful transcription of the ``horizontal_diffusion`` scalar (``ELSE``) branch
    (``module_big_step_utilities_em.F:2926-2946``) for unit map factors
    (``msf=msfvx_inv=1``).  Returns the ALREADY mass-coupled tendency
    ``d/dx(mass_f*K_f*df/dx) + d/dy(mass_f*K_f*df/dy)`` (do NOT multiply by mass
    again), where the face mass ``mass_f`` and face viscosity ``K_f`` are the
    arithmetic average of the two adjacent mass-cell values:

      ``mkrdxp = 0.5*(K(i+1)+K(i)) * 0.5*(mass(i+1)+mass(i)) * rdx``  (east face)
      ``mkrdxm = 0.5*(K(i)+K(i-1)) * 0.5*(mass(i)+mass(i-1)) * rdx``  (west face)
      ``tend  += rdx*(mkrdxp*(f(i+1)-f(i)) - mkrdxm*(f(i)-f(i-1)))`` + (y-terms)

    ``mass`` is the dry-column mass ``c1*MUT+c2`` on mass cells (WRF MUT coupling).
    ``xkmhd`` is the per-field horizontal eddy diffusivity (xkmhd for momentum,
    xkhh for heat).  All fields are ``(nz, ny, nx)`` on mass cells; periodic x/y.
    """

    rdx = 1.0 / float(dx_m)
    rdy = 1.0 / float(dy_m)

    k_e = 0.5 * (jnp.roll(xkmhd, -1, axis=2) + xkmhd)  # K at east face i+1/2
    k_w = 0.5 * (xkmhd + jnp.roll(xkmhd, 1, axis=2))   # K at west face i-1/2
    m_e = 0.5 * (jnp.roll(mass, -1, axis=2) + mass)    # mass at east face
    m_w = 0.5 * (mass + jnp.roll(mass, 1, axis=2))     # mass at west face
    mkrdxp = k_e * m_e * rdx
    mkrdxm = k_w * m_w * rdx
    tend = rdx * (
        mkrdxp * (jnp.roll(field, -1, axis=2) - field)
        - mkrdxm * (field - jnp.roll(field, 1, axis=2))
    )

    if field.shape[1] > 1:
        k_n = 0.5 * (jnp.roll(xkmhd, -1, axis=1) + xkmhd)
        k_s = 0.5 * (xkmhd + jnp.roll(xkmhd, 1, axis=1))
        m_n = 0.5 * (jnp.roll(mass, -1, axis=1) + mass)
        m_s = 0.5 * (mass + jnp.roll(mass, 1, axis=1))
        mkrdyp = k_n * m_n * rdy
        mkrdym = k_s * m_s * rdy
        tend = tend + rdy * (
            mkrdyp * (jnp.roll(field, -1, axis=1) - field)
            - mkrdym * (field - jnp.roll(field, 1, axis=1))
        )
    return tend


def horizontal_diffusion_coord_scalar_tendency(
    field: jax.Array,
    xkhh: jax.Array,
    mass: jax.Array,
    *,
    dx_m: float,
    dy_m: float,
    base_3d: jax.Array | None = None,
) -> jax.Array:
    """diff_opt=1 coordinate-surface scalar (theta) horizontal diffusion.

    Faithful transcription of ``horizontal_diffusion_3dmp``
    (``module_big_step_utilities_em.F:3033-3058``): the diffusion acts on the
    PERTURBATION ``field - base_3d`` (WRF passes ``t_init`` as ``base_3d`` so the
    diffusion does not erode the reference profile).  When ``base_3d`` is None the
    full field is diffused (== horizontal_diffusion scalar branch).  Returns the
    mass-coupled tendency.  ``xkhh`` is the HEAT eddy diffusivity (= 3*xkmh via
    prandtl=1/3, matching WRF khdq=3*khdif).
    """

    diff_field = field if base_3d is None else (field - base_3d)
    return _hdiff_coord_scalar(diff_field, xkhh, mass, dx_m=dx_m, dy_m=dy_m)


def horizontal_diffusion_coord_momentum_tendency(
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    xkmhd_h: jax.Array,
    mass_u: jax.Array,
    mass_v: jax.Array,
    mass_f: jax.Array,
    *,
    dx_m: float,
    dy_m: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """diff_opt=1 coordinate-surface horizontal diffusion of u, v, w (km_opt=4).

    WRF ``horizontal_diffusion`` ('u'/'v'/'w' branches,
    ``module_big_step_utilities_em.F:2779-2909``).  For unit map factors the three
    momentum branches collapse to the SAME variable-K mass-weighted flux divergence
    as the scalar branch -- the only WRF differences (the msf ratios and the
    four-corner / staggered K averaging) are all unity / identical-on-the-slab when
    ``msf=1`` and ``K`` varies smoothly, so the leading-order WRF-faithful operator
    on the flat periodic slab is the cell-face flux divergence applied on each
    field's own stagger.

    The momentum eddy viscosity ``xkmhd_h`` (from :func:`smag2d_horizontal_km`,
    ``xkmh``) lives on mass cells.  u lives on x-faces, v on y-faces, w on z-faces
    (over the mass-x/y columns).  Returns coupled ``(du, dv, dw)`` on each field's
    own stagger.

    NOTE: the WRF u/v branches average xkmhd to the u/v points (four-corner for the
    cross-derivative term); on the unit-msf periodic slab with the cell-centred K
    this reduces to the same face-averaged K used here to the operator's 2nd-order
    accuracy.  This is the documented idealized-slab reduction (same scope as the
    existing constant-K deformation path), NOT a tuning approximation.
    """

    nx = w.shape[-1]
    ny = w.shape[1]
    u_m = u[:, :, :nx] if u.shape[-1] == nx + 1 else u
    v_m = v[:, :ny, :] if v.shape[1] == ny + 1 else v
    mass_u_m = mass_u[:, :, :nx] if mass_u.shape[-1] == nx + 1 else mass_u
    mass_v_m = mass_v[:, :ny, :] if mass_v.shape[1] == ny + 1 else mass_v

    du = _hdiff_coord_scalar(u_m, xkmhd_h, mass_u_m, dx_m=dx_m, dy_m=dy_m)
    dv = _hdiff_coord_scalar(v_m, xkmhd_h, mass_v_m, dx_m=dx_m, dy_m=dy_m)
    # w is on z-faces (nz+1 levels) over the mass columns; diffuse horizontally
    # level-by-level with the w-level viscosity (xkmhd averaged in z to w faces).
    xkmhd_w = _xkmhd_to_w_levels(xkmhd_h, w.shape[0])
    dw = _hdiff_coord_scalar(w, xkmhd_w, mass_f, dx_m=dx_m, dy_m=dy_m)

    # restore the periodic wrap faces for u/v if the inputs carried them.
    if u.shape[-1] == nx + 1:
        du = jnp.concatenate((du, du[:, :, :1]), axis=2)
    if v.shape[1] == ny + 1:
        dv = jnp.concatenate((dv, dv[:, :1, :]), axis=1)
    return du, dv, dw


__all__ = [
    "sixth_order_diffusion_tendency",
    "constant_k_diffusion_tendency",
    "conservative_constant_k_diffusion_tendency",
    "constant_k_deformation_momentum_tendency",
    "wrf_deformation_momentum_tendency",
    "deformation_components_3d",
    "wrf_terrain_deformation_momentum_tendency",
    "PRANDTL",
    "C_S_DEFAULT",
    "horizontal_deformation_2d",
    "smag2d_horizontal_km",
    "horizontal_diffusion_coord_scalar_tendency",
    "horizontal_diffusion_coord_momentum_tendency",
]
