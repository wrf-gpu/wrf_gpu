"""WRF ``rk_tendency`` large-step PGF + ``rk_addtend_dry`` per-stage merge.

Two distinct WRF behaviours live here:

* :func:`large_step_horizontal_pgf` reproduces the large-step horizontal
  pressure-gradient force that WRF ``rk_tendency`` adds into the *coupled*
  large-step momentum tendencies ``ru_tend``/``rv_tend``
  (``dyn_em/module_em.F:1325`` calls ``horizontal_pressure_gradient`` whose body
  is ``dyn_em/module_big_step_utilities_em.F:2459-2466`` for u and ``:2379-2386``
  for v).  This term uses the **absolute** ``rk_step_prep`` diagnostics
  (``ph``, ``alt``, ``p``, ``pb``, ``al``, ``php``) and is the steady gradient
  that drives the mean circulation.  It is a *different split term* from the
  small-step ``advance_uv`` acoustic PGF (``module_small_step_em.F:828-868``),
  which uses the small-step work-array perturbation pressure that starts ~0 at
  each RK stage; the two are NOT a double-count.

* :func:`rk_addtend_dry` reproduces the WRF per-RK-stage merge of the
  RK1-fixed physics tendencies ``*_tendf`` into the per-stage dry-dynamics
  tendencies ``*_tend`` with field-specific map-factor / mass coupling
  (``dyn_em/module_em.F:1711-1786``).  u uses ``1/msfuy``, v uses ``msfvx_inv``,
  w/ph/theta use ``1/msfty``, mu is uncoupled, and the final-RK theta picks up
  the diabatic-heating term ``(c1*mut+c2)*h_diabatic/msfty``.  With physics off
  and periodic boundaries (the idealized dry gate) ``*_tendf == 0`` and the
  RK1 boundary-save adds vanish, so the merge is the identity on ``*_tend`` --
  it is implemented faithfully so the cadence is correct once physics is on.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.dynamics.acoustic_wrf import (
    _inverse_density_from_theta_pressure,
    moisture_coupling_factors,
)


_SHARDED_HALO_CONTEXT: tuple[object, int] | None = None


def _maybe_sharded_x_edge_pair(
    field: jax.Array,
    left: jax.Array,
    right: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Make x-face pairs match global-domain edge padding under opt-in x sharding."""

    context = _SHARDED_HALO_CONTEXT
    if context is None:
        return left, right
    sharding, width = context
    if not bool(getattr(sharding, "enabled", False)):
        return left, right
    if getattr(sharding, "axis", "x") != "x":
        raise NotImplementedError("large-step dry sharded edge-pair correction supports x-axis decomposition only")
    h = int(width)
    owned = int(field.shape[-1]) - 2 * h
    if owned < 1:
        raise ValueError("haloed x field has no owned cells")
    rank = jax.lax.axis_index(str(sharding.axis_name))
    start = rank * owned
    global_nx = owned * int(sharding.resolved_partitions())
    west_face = h
    east_face = h + owned
    is_first = start == 0
    is_last = start + owned == global_nx
    left = left.at[..., west_face].set(jnp.where(is_first, right[..., west_face], left[..., west_face]))
    right = right.at[..., east_face].set(jnp.where(is_last, left[..., east_face], right[..., east_face]))
    return left, right


def _x_face_pair_3d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return (left=west mass cell, right=east mass cell) on u-faces (nx+1).

    Edge-padded to match ``advance_uv_wrf``: face ``i`` sits between mass cells
    ``i-1`` (left) and ``i`` (right); boundary faces repeat the edge mass cell.
    """
    padded = jnp.pad(field, ((0, 0), (0, 0), (1, 1)), mode="edge")
    return _maybe_sharded_x_edge_pair(field, padded[:, :, :-1], padded[:, :, 1:])


def _y_face_pair_3d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    padded = jnp.pad(field, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return padded[:, :-1, :], padded[:, 1:, :]


def _x_face_pair_2d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    padded = jnp.pad(field, ((0, 0), (1, 1)), mode="edge")
    return _maybe_sharded_x_edge_pair(field, padded[:, :-1], padded[:, 1:])


def _y_face_pair_2d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    padded = jnp.pad(field, ((1, 1), (0, 0)), mode="edge")
    return padded[:-1, :], padded[1:, :]


@dataclass(frozen=True)
class DryPhysicsTendencies:
    """RK1-fixed physics + boundary tendencies (``*_tendf`` in WRF).

    All leaves default to zero (physics-off dry gate).  ``h_diabatic`` is the
    diabatic heating that enters the final-RK theta tendency
    (``module_em.F:1770-1773``).  ``*_save`` are the boundary tendencies that
    WRF adds to ``*_tendf`` only on ``rk_step == 1`` (``:1735``, ``:1746``,
    ``:1757``, ``:1760``, ``:1770``); for periodic idealized cases they are 0.
    """

    ru_tendf: jax.Array | None = None
    rv_tendf: jax.Array | None = None
    rw_tendf: jax.Array | None = None
    ph_tendf: jax.Array | None = None
    t_tendf: jax.Array | None = None
    mu_tendf: jax.Array | None = None
    h_diabatic: jax.Array | None = None
    u_save: jax.Array | None = None
    v_save: jax.Array | None = None
    w_save: jax.Array | None = None
    ph_save: jax.Array | None = None
    t_save: jax.Array | None = None


def _absolute_diagnostics(
    state: State, metrics: DycoreMetrics, *, t0: float = 300.0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return WRF ``rk_step_prep`` diagnostics ``(ph', p', al', alt, php)``.

    These feed the large-step *horizontal* PGF (WRF ``rk_tendency`` ->
    ``horizontal_pressure_gradient``, ``module_em.F:1325``).  WRF builds them in
    ``rk_step_prep``/``calc_p_rho_phi`` (``module_em.F:184-225``;
    ``module_big_step_utilities_em.F:1023-1030``) from the *state* perturbation
    fields, NOT by re-deriving a synthetic pressure from absolute theta:

      ``p'``   = ``grid%p`` = ``state.p_perturbation`` (the WRF perturbation
                 pressure diagnostic; this is the field passed to the PGF, not a
                 second theta-derived pressure).  F7F: previously this re-derived
                 ``p_abs`` from absolute θ, inventing a vertical pressure source
                 that double-counted the buoyancy; removed.
      ``al'``  = ``-1/(c1h*muts+c2h) * (alb*c1h*mu' + rdnw*(ph'(k+1)-ph'(k)))``
                 from the perturbation geopotential ``ph'`` and ``mu'``
                 (WRF ``calc_p_rho_phi``, ``module_big_step_utilities_em.F:1029``).
                 The denominator is the TOTAL column mass ``muts`` (WRF passes
                 ``grid%muts = grid%mut + grid%mu_2``, solve_em.F:3610) and the
                 ``mu'`` term carries the BASE-state inverse density ``alb`` --
                 NOT the prior JAX form which used the base column mass ``mub`` in
                 the denominator and the TOTAL inverse density ``alt`` on the
                 ``mu'`` term.  Opus v040 round (2026-06-03) adjudicated this
                 against agy's review: it is a REAL correctness bug but a small one
                 (al relative error ~1.2e-4 of alt; <0.01 m/s of the 24h U10 bias).
                 Fixed here for WRF-faithfulness regardless.
      ``alb``  = base-state inverse density from the hydrostatic base column,
                 ``alb = -rdnw*(phb(k+1)-phb(k)) / (c1h*mub+c2h)`` (the exact
                 inverse of the base hydrostatic relation; WRF carries ``grid%alb``
                 from init, identical to this reconstruction at machine precision).
      ``alt``  = full inverse density from the EOS (θ_total, p_total).
      ``php``  = ``0.5*(phb+ph' faces)`` on mass levels (full geopotential).
    """

    ph_pert = state.ph_perturbation.astype(jnp.float64)
    mu_pert = state.mu_perturbation.astype(jnp.float64)
    mu_total = state.mu_total.astype(jnp.float64)
    mub = (state.mu_total - state.mu_perturbation).astype(jnp.float64)
    alt = _inverse_density_from_theta_pressure(
        state.theta.astype(jnp.float64), state.p_total.astype(jnp.float64)
    )
    # WRF p for the horizontal PGF is grid%p = the perturbation-pressure
    # diagnostic carried on the state, not a re-derived absolute-θ pressure.
    p_pert = state.p_perturbation.astype(jnp.float64)
    c1h = metrics.c1h[:, None, None]
    c2h = metrics.c2h[:, None, None]
    rdnw = metrics.rdnw[:, None, None]
    phb = (state.ph_total - state.ph_perturbation).astype(jnp.float64)
    # WRF-faithful al (calc_p_rho_phi :1029).  Denominator = TOTAL column mass
    # muts = mut + mu_2 (solve_em.F:3610); the mu' term carries the BASE-state
    # inverse density alb (reconstructed from the base hydrostatic column).
    mass_base = c1h * mub[None, :, :] + c2h
    safe_base = jnp.where(jnp.abs(mass_base) > 1.0e-12, mass_base, jnp.asarray(1.0e-12, dtype=mass_base.dtype))
    alb = -rdnw * (phb[1:, :, :] - phb[:-1, :, :]) / safe_base
    muts = mu_total + mu_pert  # grid%muts = grid%mut + grid%mu_2
    mass_t = c1h * muts[None, :, :] + c2h
    safe_t = jnp.where(jnp.abs(mass_t) > 1.0e-12, mass_t, jnp.asarray(1.0e-12, dtype=mass_t.dtype))
    mu_term = c1h * mu_pert[None, :, :]
    al = -(alb * mu_term + rdnw * (ph_pert[1:, :, :] - ph_pert[:-1, :, :])) / safe_t
    ph_total = phb + ph_pert
    php = 0.5 * (ph_total[:-1, :, :] + ph_total[1:, :, :])
    return ph_pert, p_pert, al, alt, php


def large_step_horizontal_pgf(
    state: State,
    metrics: DycoreMetrics,
    *,
    dx_m: float,
    dy_m: float,
    non_hydrostatic: bool = True,
    top_lid: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return the WRF large-step *coupled* horizontal PGF for ``ru/rv_tend``.

    Source: WRF ``rk_tendency`` -> ``horizontal_pressure_gradient``
    (``module_em.F:1325``; body ``module_big_step_utilities_em.F:2453-2488`` for
    u, ``:2373-2404`` for v).  WRF adds ``ru_tend -= cqu*dpx`` where ``dpx``
    carries the coupled ``c1h*muu+c2h`` mass factor, so the returned tendency is
    the **coupled** momentum tendency consumed by ``advance_uv`` (``u +=
    dts*ru_tend``, ``module_small_step_em.F:805``).

    Uses the *absolute* ``rk_step_prep`` diagnostics (full geopotential ``php``,
    full inverse density ``alt``, absolute perturbation pressure ``p'``, base
    pressure ``pb``, perturbation inverse density ``al'``) so the steady gradient
    from the bubble's hydrostatic imbalance is captured -- this is the
    DISTINCT split term from the small-step acoustic PGF in ``advance_uv``, which
    uses the work-array perturbation pressure that restarts ~0 each RK stage.

    C-grid (periodic): the u-face at index ``i`` lies between mass cells ``i-1``
    and ``i``; ``dpx(i)`` differences ``field(i)-field(i-1)``.  The 2dx slab has a
    singleton y axis, so the v-PGF is structurally zero there.
    """

    ph, p_abs, al, alt, php = _absolute_diagnostics(state, metrics)
    pb = (state.p_total - state.p_perturbation).astype(jnp.float64)
    mut = (state.mu_total - state.mu_perturbation).astype(jnp.float64)
    mu_pert = state.mu_perturbation.astype(jnp.float64)
    cqu, cqv = moisture_coupling_factors(state)
    rdx = 1.0 / float(dx_m)
    rdy = 1.0 / float(dy_m)

    c1h = metrics.c1h[:, None, None]
    c2h = metrics.c2h[:, None, None]
    rdnw = metrics.rdnw[:, None, None]
    msf_u = (metrics.msfux / metrics.msfuy)[None, :, :]
    msf_v = (metrics.msfvy / metrics.msfvx)[None, :, :]
    mu_total = mut + mu_pert

    # WRF top-face (lid) extrapolation coefficients cfn/cfn1 for dpn(kde) when
    # top_lid is active (module_big_step_utilities_em.F:2357-2362 uses cfn/cfn1 --
    # NOT the bottom cf1/cf2/cf3).  WRF init (e.g. module_initialize_scm_xy.F:351):
    #   cfn  = (0.5*dnw[kde-1] + dn[kde-1]) / dn[kde-1]
    #   cfn1 = -0.5*dnw[kde-1] / dn[kde-1]
    # with kde-1 the top mass level (JAX index nz-1).  Opus v040 round (2026-06-03):
    # agy flagged the prior JAX form (cf1/cf2/cf3 at the top) as a real but tiny bug
    # active only on the top model level; fixed here for WRF-faithfulness (top-face
    # only, ~1e-8 m/s of the 24h surface wind bias).
    _dn_top = metrics.dn[-1]
    _dn_top_safe = jnp.where(jnp.abs(_dn_top) > 1.0e-30, _dn_top, jnp.asarray(1.0, dtype=_dn_top.dtype))
    cfn = (0.5 * metrics.dnw[-1] + metrics.dn[-1]) / _dn_top_safe
    cfn1 = -0.5 * metrics.dnw[-1] / _dn_top_safe

    def _dpn_faces(pair_sum: jax.Array) -> jax.Array:
        nz = int(pair_sum.shape[0])
        dpn = jnp.zeros((nz + 1,) + pair_sum.shape[1:], dtype=pair_sum.dtype)
        bottom = 0.5 * (metrics.cf1 * pair_sum[0] + metrics.cf2 * pair_sum[1] + metrics.cf3 * pair_sum[2])
        dpn = dpn.at[0, :, :].set(bottom)
        interior = 0.5 * (
            metrics.fnm[1:, None, None] * pair_sum[1:, :, :] + metrics.fnp[1:, None, None] * pair_sum[:-1, :, :]
        )
        dpn = dpn.at[1:nz, :, :].set(interior)
        if bool(top_lid):
            top = 0.5 * (cfn * pair_sum[-1] + cfn1 * pair_sum[-2])
            dpn = dpn.at[nz, :, :].set(top)
        return dpn

    # --- x PGF (WRF module_big_step_utilities_em.F:2459-2466) on u-faces (nx+1) ---
    ph_l, ph_r = _x_face_pair_3d(ph)
    p_l, p_r = _x_face_pair_3d(p_abs)
    pb_l, pb_r = _x_face_pair_3d(pb)
    al_l, al_r = _x_face_pair_3d(al)
    alt_l, alt_r = _x_face_pair_3d(alt)
    muu = _x_face_pair_2d(mu_total)
    muu = 0.5 * (muu[0] + muu[1])
    mass_u = c1h * muu[None, :, :] + c2h
    ph_term_x = (ph_r[1:, :, :] - ph_l[1:, :, :]) + (ph_r[:-1, :, :] - ph_l[:-1, :, :])
    p_term_x = (alt_l + alt_r) * (p_r - p_l)
    pb_term_x = (al_l + al_r) * (pb_r - pb_l)
    dpx = msf_u * 0.5 * rdx * mass_u * (ph_term_x + p_term_x + pb_term_x)
    if bool(non_hydrostatic):
        php_l, php_r = _x_face_pair_3d(php)
        psum_l, psum_r = _x_face_pair_3d(p_abs)
        dpn = _dpn_faces(psum_l + psum_r)
        mu_l, mu_r = _x_face_pair_2d(mu_pert)
        bracket = rdnw * (dpn[1:, :, :] - dpn[:-1, :, :]) - 0.5 * (c1h * (mu_l + mu_r)[None, :, :])
        dpx = dpx + msf_u * rdx * (php_r - php_l) * bracket

    # --- y PGF (WRF module_big_step_utilities_em.F:2379-2386) on v-faces (ny+1) ---
    ph_s, ph_n = _y_face_pair_3d(ph)
    p_s, p_n = _y_face_pair_3d(p_abs)
    pb_s, pb_n = _y_face_pair_3d(pb)
    al_s, al_n = _y_face_pair_3d(al)
    alt_s, alt_n = _y_face_pair_3d(alt)
    muv = _y_face_pair_2d(mu_total)
    muv = 0.5 * (muv[0] + muv[1])
    mass_v = c1h * muv[None, :, :] + c2h
    ph_term_y = (ph_n[1:, :, :] - ph_s[1:, :, :]) + (ph_n[:-1, :, :] - ph_s[:-1, :, :])
    p_term_y = (alt_s + alt_n) * (p_n - p_s)
    pb_term_y = (al_s + al_n) * (pb_n - pb_s)
    dpy = msf_v * 0.5 * rdy * mass_v * (ph_term_y + p_term_y + pb_term_y)
    if bool(non_hydrostatic):
        php_s, php_n = _y_face_pair_3d(php)
        psum_s, psum_n = _y_face_pair_3d(p_abs)
        dpn_y = _dpn_faces(psum_s + psum_n)
        mu_s, mu_n = _y_face_pair_2d(mu_pert)
        bracket_y = rdnw * (dpn_y[1:, :, :] - dpn_y[:-1, :, :]) - 0.5 * (c1h * (mu_s + mu_n)[None, :, :])
        dpy = dpy + msf_v * rdy * (php_n - php_s) * bracket_y

    ru_pgf = -cqu * dpx
    rv_pgf = -cqv * dpy
    return ru_pgf, rv_pgf


def large_step_coriolis(
    state: State,
    metrics: DycoreMetrics,
    *,
    specified: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Return the WRF large-step *coupled* Coriolis tendency for ``ru/rv_tend``.

    Source: WRF ``rk_tendency`` -> ``coriolis``
    (``module_em.F:761``; body ``module_big_step_utilities_em.F:3640-3850``;
    ``config_flags%pert_coriolis`` defaults to ``.false.`` (``Registry.EM_COMMON``)
    so the Canary real case uses the standard ``coriolis``).  WRF assembles this
    term in the SAME coupled ``ru/rv_tend`` space as the PGF and consumes it inside
    ``advance_uv`` (``u += dts*ru_tend``, ``module_small_step_em.F:805``), so the
    returned tendency is added to the PGF before ``rk_addtend_dry`` exactly as the
    Fortran does (``module_em.F:717`` PGF then ``:761`` coriolis).

    Coupled momentum (``couple_momentum``, ``module_big_step_utilities_em.F:372/383``):
    ``ru = u*(c1h*muu+c2h)/msfuy`` on u-faces, ``rv = v*(c1h*muv+c2h)/msfvx`` on
    v-faces, ``rw = w*(c1f*mut+c2f)/msfty`` on w-faces.

    C-grid staggering (the key correctness risk): ``f``/``e``/``sina``/``cosa`` live
    on mass points; they are averaged onto the u-face (``0.5*(f(i)+f(i-1))``,
    :3726) and v-face (``0.5*(f(j)+f(j-1))``, :3800).  The off-axis coupled momentum
    is averaged from its four surrounding faces onto the target face: ``rv`` over the
    x-pair ``(i-1,i)`` and y-pair ``(j,j+1)`` for the u-eqn (:3727); ``ru`` over the
    x-pair ``(i,i+1)`` and y-pair ``(j-1,j)`` for the v-eqn (:3801).  Sign: ``+f*rv``
    into ``ru_tend`` (:3726), ``-f*ru`` into ``rv_tend`` (:3800).

    The cosine-Coriolis ``e*rw`` and the ``sina/cosa`` map-rotation pieces are kept
    (they enter the *horizontal* tendencies only, never ``rw_tend``/the w-solve);
    at the Canary domain ``e~1.3e-4`` against ``w~O(0.05)`` makes them ~1% of the
    ``f`` term and ``sina~0.02`` the rotation ~2%, but they are WRF-faithful and the
    leaves default to ``e=0,sina=0,cosa=1`` for idealized cases.  The vertical-
    momentum Coriolis (``rw_tend += e*ru``, :3839) is intentionally NOT applied here
    (it feeds the acoustic w-phi solve, which is out of scope).

    ``specified``: WRF skips the outermost u-face column / v-face row for
    specified/nested boundaries (``i_start=MAX(ids+1,its)`` etc., :3714/:3776).
    The Canary real case is nested (``specified=True``); the excluded edge faces are
    overwritten by the lateral-boundary relaxation anyway, so zeroing Coriolis there
    keeps the interior identical to WRF without perturbing the boundary frame.  For
    idealized cases ``f=0`` makes every term identically zero regardless.
    """

    u = jnp.asarray(state.u, dtype=jnp.float64)  # (nz, ny, nx+1)
    v = jnp.asarray(state.v, dtype=jnp.float64)  # (nz, ny+1, nx)
    w = jnp.asarray(state.w, dtype=jnp.float64)  # (nz+1, ny, nx)
    mu_total = jnp.asarray(state.mu_total, dtype=jnp.float64)  # (ny, nx)

    c1h = metrics.c1h[:, None, None]
    c2h = metrics.c2h[:, None, None]
    c1f = metrics.c1f[:, None, None]
    c2f = metrics.c2f[:, None, None]

    # --- coupled momentum (couple_momentum, :372/383/394) ---
    muu = 0.5 * sum(_x_face_pair_2d(mu_total))  # (ny, nx+1)
    muv = 0.5 * sum(_y_face_pair_2d(mu_total))  # (ny+1, nx)
    ru = u * (c1h * muu[None, :, :] + c2h) / metrics.msfuy[None, :, :]
    rv = v * (c1h * muv[None, :, :] + c2h) / metrics.msfvx[None, :, :]
    rw = w * (c1f * mu_total[None, :, :] + c2f) / metrics.msfty[None, :, :]

    # f/e/sina/cosa on mass points -> staggered face averages.
    f_u = 0.5 * sum(_x_face_pair_2d(metrics.f))  # (ny, nx+1)
    e_u = 0.5 * sum(_x_face_pair_2d(metrics.e))
    cosa_u = 0.5 * sum(_x_face_pair_2d(metrics.cosa))
    f_v = 0.5 * sum(_y_face_pair_2d(metrics.f))  # (ny+1, nx)
    e_v = 0.5 * sum(_y_face_pair_2d(metrics.e))
    sina_v = 0.5 * sum(_y_face_pair_2d(metrics.sina))

    msf_u = (metrics.msfux / metrics.msfuy)[None, :, :]  # (1, ny, nx+1)
    msf_v = (metrics.msfvy / metrics.msfvx)[None, :, :]  # (1, ny+1, nx)

    # === u-momentum coriolis (:3726-3729) on u-faces (nz, ny, nx+1) ===
    # rv averaged from the four surrounding v-faces: x-pair (i-1,i), y-pair (j,j+1).
    # rv is (nz, ny+1, nx); pad x by edge to reach the nx+1 u-faces, take the j and
    # j+1 v-face rows, then average the (i-1,i) and (j,j+1) quad.
    rv_xpad = jnp.pad(rv, ((0, 0), (0, 0), (1, 1)), mode="edge")  # (nz, ny+1, nx+2)
    rv_im1 = rv_xpad[:, :, :-1]  # west neighbour for each u-face (nz, ny+1, nx+1)
    rv_i = rv_xpad[:, :, 1:]  # east neighbour for each u-face (nz, ny+1, nx+1)
    rv_im1, rv_i = _maybe_sharded_x_edge_pair(rv, rv_im1, rv_i)
    rv_quad = 0.25 * (rv_im1[:, :-1, :] + rv_i[:, :-1, :] + rv_im1[:, 1:, :] + rv_i[:, 1:, :])
    ru_cor = msf_u * f_u[None, :, :] * rv_quad

    # cosine-coriolis -e*cosa*0.25*(rw quad over x-pair (i-1,i), z-pair (k,k+1)).
    rw_xpad = jnp.pad(rw, ((0, 0), (0, 0), (1, 1)), mode="edge")  # (nz+1, ny, nx+2)
    rw_im1 = rw_xpad[:, :, :-1]  # (nz+1, ny, nx+1)
    rw_i = rw_xpad[:, :, 1:]
    rw_im1, rw_i = _maybe_sharded_x_edge_pair(rw, rw_im1, rw_i)
    rw_u_quad = 0.25 * (rw_im1[:-1] + rw_im1[1:] + rw_i[:-1] + rw_i[1:])  # (nz, ny, nx+1)
    ru_cor = ru_cor - e_u[None, :, :] * cosa_u[None, :, :] * rw_u_quad

    if specified:
        # WRF excludes the first/last u-face column for specified/nested (:3714-3717).
        edge_mask_u = jnp.ones((1, 1, u.shape[2]), dtype=jnp.float64)
        edge_mask_u = edge_mask_u.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        ru_cor = ru_cor * edge_mask_u

    # === v-momentum coriolis (:3800-3803) on v-faces (nz, ny+1, nx) ===
    # ru averaged from the four surrounding u-faces: x-pair (i,i+1), y-pair (j-1,j).
    ru_ypad = jnp.pad(ru, ((0, 0), (1, 1), (0, 0)), mode="edge")  # (nz, ny+2, nx+1)
    ru_jm1 = ru_ypad[:, :-1, :]  # south neighbour for each v-face (nz, ny+1, nx+1)
    ru_j = ru_ypad[:, 1:, :]  # north neighbour for each v-face (nz, ny+1, nx+1)
    ru_quad = 0.25 * (ru_jm1[:, :, :-1] + ru_jm1[:, :, 1:] + ru_j[:, :, :-1] + ru_j[:, :, 1:])
    rv_cor = -msf_v * f_v[None, :, :] * ru_quad

    # cosine-coriolis +msf_v*e*sina*0.25*(rw quad over y-pair (j-1,j), z-pair (k,k+1)).
    rw_ypad = jnp.pad(rw, ((0, 0), (1, 1), (0, 0)), mode="edge")  # (nz+1, ny+2, nx)
    rw_jm1 = rw_ypad[:, :-1, :]  # (nz+1, ny+1, nx)
    rw_j = rw_ypad[:, 1:, :]
    rw_v_quad = 0.25 * (rw_jm1[:-1] + rw_jm1[1:] + rw_j[:-1] + rw_j[1:])  # (nz, ny+1, nx)
    rv_cor = rv_cor + msf_v * e_v[None, :, :] * sina_v[None, :, :] * rw_v_quad

    if specified:
        # WRF excludes the first/last v-face row for specified/nested (:3776-3778).
        edge_mask_v = jnp.ones((1, v.shape[1], 1), dtype=jnp.float64)
        edge_mask_v = edge_mask_v.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
        rv_cor = rv_cor * edge_mask_v

    return ru_cor, rv_cor


def _zeros_like(reference: jax.Array, candidate: jax.Array | None) -> jax.Array:
    return jnp.zeros_like(reference) if candidate is None else jnp.asarray(candidate, dtype=reference.dtype)


def rk_addtend_dry(
    tendencies: Tendencies,
    physics: DryPhysicsTendencies,
    *,
    rk_step: int,
    metrics: DycoreMetrics,
    mut: jax.Array,
) -> Tendencies:
    """Merge RK1-fixed physics tendencies into per-stage dry tendencies.

    Source: WRF ``dyn_em/module_em.F:1711-1786`` (subroutine ``rk_addtend_dry``).
    Field-specific coupling (WRF comments ``:1722-1729``):

    * u: ``ru_tend += (ru_tendf + [rk1] u_save*msfuy)/msfuy``  (``:1735-1737``)
    * v: ``rv_tend += (rv_tendf + [rk1] v_save*msfvx)*msfvx_inv``  (``:1746-1748``)
    * w: ``rw_tend += (rw_tendf + [rk1] w_save*msfty)/msfty``  (``:1757-1759``)
    * ph: ``ph_tend += (ph_tendf + [rk1] ph_save)/msfty``  (``:1760-1762``)
    * theta: ``t_tend += (t_tendf + [rk1] t_save)/msfty + (c1*mut+c2)*h_diabatic/msfty``  (``:1770-1773``)
    * mu: ``mu_tend += mu_tendf``  (uncoupled, ``:1782``)

    ``tendencies`` carries the per-stage dry dynamics (advection + large-step PGF
    + diffusion), in the same uncoupled / coupled convention the operational RK
    path already uses for ``advance_uv``/``advance_mu_t``/``advance_w``.  With
    physics off and periodic boundaries every ``*_tendf`` and ``*_save`` is zero,
    so this returns ``tendencies`` unchanged numerically.
    """

    one = jnp.asarray(1.0, dtype=tendencies.u.dtype)
    msfuy = metrics.msfuy[None, :, :]
    msfvx_inv = (one / metrics.msfvx)[None, :, :]
    msfvx = metrics.msfvx[None, :, :]
    msfty = metrics.msfty[None, :, :]

    ru_tendf = _zeros_like(tendencies.u, physics.ru_tendf)
    rv_tendf = _zeros_like(tendencies.v, physics.rv_tendf)
    rw_tendf = _zeros_like(tendencies.w, physics.rw_tendf)
    ph_tendf = _zeros_like(tendencies.ph, physics.ph_tendf)
    t_tendf = _zeros_like(tendencies.theta, physics.t_tendf)
    mu_tendf = _zeros_like(tendencies.mu, physics.mu_tendf)

    if int(rk_step) == 1:
        ru_tendf = ru_tendf + _zeros_like(tendencies.u, physics.u_save) * msfuy
        rv_tendf = rv_tendf + _zeros_like(tendencies.v, physics.v_save) * msfvx
        rw_tendf = rw_tendf + _zeros_like(tendencies.w, physics.w_save) * msfty
        ph_tendf = ph_tendf + _zeros_like(tendencies.ph, physics.ph_save)
        t_tendf = t_tendf + _zeros_like(tendencies.theta, physics.t_save)

    u = tendencies.u + ru_tendf / msfuy
    v = tendencies.v + rv_tendf * msfvx_inv
    w = tendencies.w + rw_tendf / msfty
    ph = tendencies.ph + ph_tendf / msfty
    mass_h = metrics.c1h[:, None, None] * mut[None, :, :] + metrics.c2h[:, None, None]
    h_diabatic = _zeros_like(tendencies.theta, physics.h_diabatic)
    theta = tendencies.theta + t_tendf / msfty + mass_h * h_diabatic / msfty
    mu = tendencies.mu + mu_tendf

    return tendencies.replace(u=u, v=v, w=w, ph=ph, theta=theta, mu=mu)


__all__ = ["DryPhysicsTendencies", "large_step_horizontal_pgf", "rk_addtend_dry"]
