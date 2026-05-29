"""WRF-faithful large-step geopotential-equation RHS ``rhs_ph`` -> ``ph_tend``.

Source: WRF ``dyn_em/module_big_step_utilities_em.F:1365-2232`` (subroutine
``rhs_ph``), called once per RK stage from ``rk_tendency``
(``dyn_em/module_em.F:1254-1266``) with the stage explicit omega ``wwE``
(``= grid%ww``, ``solve_em.F:762``) and the stage geopotential perturbation
``ph``.

``rhs_ph`` builds the large-timestep tendency of the geopotential equation,
cast in advective (non-flux) form (``module_big_step_utilities_em.F:1433-1439``):

    mu/my d/dt(phi) = -(1/my) mx mu u d/dx(phi)         (term 1, x advection)
                      -(1/my) my mu v d/dy(phi)         (term 2, y advection)
                      - omega d/d_eta(phi)              (term 3, vertical adv.)
                      + mu g w / my                     (term 4, "gw")

The result ``ph_tend`` is the *coupled* (``mu/my``-weighted) large-step
tendency; ``advance_w`` decouples it when it folds ``ph_tend`` into the
implicit w/phi acoustic solve (``module_small_step_em.F:1345``,
``module_big_step_utilities_em.F:1472-1477``).

Cadence note (WRF time-split design, NOT a double count): WRF zeroes
``ph_tend`` (``module_em.F:651``) and ``rhs_ph`` accumulates *all four* terms
using the *stage* (large-step, frozen) omega ``wwE`` and the *stage*
geopotential.  ``advance_w`` then adds its own term-3 / term-4 contributions
using the *small-step evolving* omega ``ww`` and the *reference* geopotential
``ph_1`` (``module_small_step_em.F:1357-1382``, ``:1345``, ``:1583``).  The two
sets use different (large-step vs small-step) fields, so both are required;
this matches the acoustic time-splitting and is exactly what the JAX
``advance_w_wrf`` already does for its small-step half.

Array layout (JAX): leading axis is the vertical index.  Faces have shape
``(nz+1, ny, nx)`` and mass levels ``(nz, ny, nx)``.  ``ph``/``phb``/``w``/``ww``
live on faces; ``u`` is x-staggered ``(nz, ny, nx+1)``, ``v`` y-staggered
``(nz, ny+1, nx)``.  WRF Fortran face index ``k`` runs ``1..kde`` (``kde=nz+1``);
the Python ``k_face`` runs ``0..nz``.

Idealized/periodic scope: unit map factors, ``phi_adv_z == 1`` (original
vertical-advection option), ``non_hydrostatic == True``, and 2nd-order
horizontal advection (``advective_order <= 2``;
``module_big_step_utilities_em.F:1546-1612``).  Horizontal differences use
periodic wrap to match the idealized doubly-periodic gate.  Map factors and
the higher-order horizontal branches are deferred (they are unit / unused on
this gate); the vertical (term 3) and buoyancy (term 4) terms — which close
the w/phi restoring loop — are full-fidelity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

GRAVITY_M_S2 = 9.81  # WRF ``g`` (share/module_model_constants.F).


def _roll_x(field: jax.Array, shift: int) -> jax.Array:
    """Periodic roll along the x (last) axis."""

    return jnp.roll(field, shift, axis=-1)


def _roll_y(field: jax.Array, shift: int) -> jax.Array:
    """Periodic roll along the y (middle) axis."""

    return jnp.roll(field, shift, axis=-2)


def rhs_ph_wrf(
    *,
    u: jax.Array,
    v: jax.Array,
    ww: jax.Array,
    ph: jax.Array,
    phb: jax.Array,
    w: jax.Array,
    mut: jax.Array,
    muu: jax.Array,
    muv: jax.Array,
    c1f: jax.Array,
    c2f: jax.Array,
    fnm: jax.Array,
    fnp: jax.Array,
    rdnw: jax.Array,
    rdx: float,
    rdy: float,
    msfty: jax.Array,
    non_hydrostatic: bool = True,
    gravity: float = GRAVITY_M_S2,
    include_vertical_gw: bool = True,
) -> jax.Array:
    """Return the coupled large-step geopotential tendency ``ph_tend``.

    ``ph`` is the stage geopotential *perturbation* on faces; ``phb`` the base
    geopotential; ``ww`` the stage (large-step) omega on faces; ``w`` the stage
    vertical velocity on faces; ``mut`` the full dry-air column mass; ``muu``,
    ``muv`` the u/v-face dry masses.  Returns ``ph_tend`` shaped like ``ph``
    (``(nz+1, ny, nx)``), nonzero on interior faces ``1..nz`` (WRF ``k=2..kde``);
    the surface face ``k=1`` (Python 0) stays zero (the lower BC is handled by
    ``advance_w``).

    WRF source: ``module_big_step_utilities_em.F:1481-1612`` (terms 3, 4, 1, 2).
    """

    nz = int(ph.shape[0]) - 1  # mass levels; faces 0..nz.
    g = float(gravity)
    ph_tend = jnp.zeros_like(ph)

    ph_total = ph + phb  # full geopotential on faces (nz+1, ny, nx)
    msft_inv = (1.0 / msfty)[None, :, :]  # (1, ny, nx)

    # ----- RHS term 3: -omega * d/d_eta(phi)  (phi_adv_z == 1, original) -----
    # WRF :1500-1518 (ELSE branch): destagger omega and multiply with the
    # mass-level partial d/dnu(phi), then re-stagger the product back to faces.
    #   wdwn(i,k) = .5*(ww(k)+ww(k-1))*rdnw(k-1)*(ph(k)-ph(k-1)+phb(k)-phb(k-1))
    #   for k=2..kte  (WRF face index)  -- this is a MASS-level quantity.
    #   ph_tend(k) -= fnm(k)*wdwn(k+1)+fnp(k)*wdwn(k)  for k=2..kte-1.
    # JAX mass-level index m=0..nz-1 corresponds to WRF k=m+1; wdwn at WRF face
    # k (2..kte) -> JAX face index k-1 = m for m=1..nz-1.  We store wdwn on the
    # mass-level grid (nz,) indexed so that wdwn_m[m] = WRF wdwn(i, m+1).
    dphi_mass = ph_total[1:, :, :] - ph_total[:-1, :, :]  # (nz, ny, nx) phi(k)-phi(k-1)
    ww_destag = 0.5 * (ww[1:, :, :] + ww[:-1, :, :])  # (nz, ny, nx) face-avg per mass level
    wdwn_mass = ww_destag * rdnw[:, None, None] * dphi_mass  # (nz, ny, nx)
    # WRF: ph_tend(k) -= fnm(k)*wdwn(k+1)+fnp(k)*wdwn(k) for faces k=2..kte-1.
    # In JAX face indexing, interior faces are f=1..nz-1; wdwn(WRF k+1)=wdwn_mass[f],
    # wdwn(WRF k)=wdwn_mass[f-1].
    if nz >= 2 and bool(include_vertical_gw):
        term3 = fnm[1:nz, None, None] * wdwn_mass[1:nz, :, :] + fnp[1:nz, None, None] * wdwn_mass[0 : nz - 1, :, :]
        ph_tend = ph_tend.at[1:nz, :, :].add(-term3)

    # ----- RHS term 4: + (c1f*mut+c2f) * g * w / my   ("gw", non-hydrostatic) -----
    # WRF :1522-1540.  ph_tend(kde)=0 (top), then for k=2..kte add the gw term.
    if bool(non_hydrostatic) and bool(include_vertical_gw):
        mass_f = c1f[:, None, None] * mut[None, :, :] + c2f[:, None, None]  # (nz+1, ny, nx)
        gw = mass_f * g * w * msft_inv  # (nz+1, ny, nx)
        # Faces k=2..kte (WRF) -> JAX faces 1..nz; top face (k=kde, JAX nz) is set
        # to zero by WRF (:1527) BEFORE the gw loop adds k=2..kte, so the gw term
        # is applied to faces 1..nz EXCEPT the very top which WRF zeroes first and
        # then the k=2..kte loop (kte=kde-1) does NOT touch face kde.  Net: gw on
        # faces 1..nz-1 (WRF k=2..kte), and face kde stays 0.
        ph_tend = ph_tend.at[1:nz, :, :].add(gw[1:nz, :, :])
        ph_tend = ph_tend.at[nz, :, :].set(0.0)

    # ----- RHS terms 1 & 2: horizontal advection of phi (2nd order) -----
    # WRF :1558-1612 (advective_order <= 2), unit map factors, periodic.
    #   y (v) advection, interior k=2..kte-1 (faces 1..nz-1):
    #     ph_tend(i,k,j) -= 0.25*rdy/my * (
    #        (c1f*muvf(j+1)+c2f)*(v(k,j+1)+v(k-1,j+1))*(phb(j+1)-phb(j)+ph(j+1)-ph(j))
    #       +(c1f*muvf(j  )+c2f)*(v(k,j  )+v(k-1,j  ))*(phb(j  )-phb(j-1)+ph(j  )-ph(j-1)) )
    #   x (u) advection symmetric with rdx and muuf(i+1)/muuf(i).
    # phi lives on faces; v/u carry one mass level below (v(k)+v(k-1)) at face k.
    if nz >= 2:
        c1f_int = c1f[1:nz, None, None]  # (nz-1, 1, 1) faces 1..nz-1

        # --- y (v) advection (term 2) ---
        # v: (nz, ny+1, nx) staggered on y.  For interior face f (1..nz-1), use
        # v at mass levels f and f-1 -> v(k)+v(k-1) with WRF k=f+1, k-1=f.
        # Wait: WRF face k uses v(i,k,j)+v(i,k-1,j); k is the FACE index, but v is
        # on mass levels.  WRF v(i,k,j) at face-loop index k means mass level k.
        # For JAX interior face f, the two mass levels straddling it are f (above)
        # and f-1 (below); WRF v(k)+v(k-1) -> v_mass[f]+v_mass[f-1].
        v_above = v[1:nz, :, :]  # mass levels 1..nz-1, shape (nz-1, ny+1, nx)
        v_below = v[0 : nz - 1, :, :]  # mass levels 0..nz-2
        v_pair = v_above + v_below  # (nz-1, ny+1, nx) on y-faces
        # v-face dry mass muv is (ny+1, nx); c1f*muv+c2f per face level.
        muvf = (c1f_int * muv[None, :, :] + c2f[1:nz, None, None])  # (nz-1, ny+1, nx)
        # ph(j+1)-ph(j) at the north v-face; ph(j)-ph(j-1) at the south v-face.
        ph_total_int = ph_total[1:nz, :, :]  # (nz-1, ny, nx) faces 1..nz-1
        dphi_north = _roll_y(ph_total_int, -1) - ph_total_int  # ph(j+1)-ph(j)
        dphi_south = ph_total_int - _roll_y(ph_total_int, 1)  # ph(j)-ph(j-1)
        # north v-face is index j+1 (mass cell j's north edge); south is index j.
        # v_pair / muvf are on (ny+1) y-faces; for mass cell j the north face is
        # v_pair[:, j+1, :] and south is v_pair[:, j, :].
        v_north = v_pair[:, 1:, :]  # (nz-1, ny, nx) north faces of each mass cell
        v_south = v_pair[:, :-1, :]  # (nz-1, ny, nx) south faces
        muvf_north = muvf[:, 1:, :]
        muvf_south = muvf[:, :-1, :]
        flux_y = muvf_north * v_north * dphi_north + muvf_south * v_south * dphi_south
        ph_tend = ph_tend.at[1:nz, :, :].add(-(0.25 * float(rdy) * msft_inv) * flux_y)

        # --- x (u) advection (term 1) ---
        # u: (nz, ny, nx+1) staggered on x.  Same vertical pairing.
        u_above = u[1:nz, :, :]
        u_below = u[0 : nz - 1, :, :]
        u_pair = u_above + u_below  # (nz-1, ny, nx+1)
        muuf = (c1f_int * muu[None, :, :] + c2f[1:nz, None, None])  # (nz-1, ny, nx+1)
        dphi_east = _roll_x(ph_total_int, -1) - ph_total_int  # ph(i+1)-ph(i)
        dphi_west = ph_total_int - _roll_x(ph_total_int, 1)  # ph(i)-ph(i-1)
        u_east = u_pair[:, :, 1:]  # east face of each mass cell (index i+1)
        u_west = u_pair[:, :, :-1]  # west face (index i)
        muuf_east = muuf[:, :, 1:]
        muuf_west = muuf[:, :, :-1]
        flux_x = muuf_east * u_east * dphi_east + muuf_west * u_west * dphi_west
        ph_tend = ph_tend.at[1:nz, :, :].add(-(0.25 * float(rdx) * msft_inv) * flux_x)

    return ph_tend


__all__ = ["rhs_ph_wrf", "GRAVITY_M_S2"]
