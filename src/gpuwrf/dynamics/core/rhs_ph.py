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

Two horizontal-advection paths:

* ``advective_order <= 2`` and ``not specified`` (the idealized/periodic gate
  path, byte-for-byte unchanged): unit map factors, 2nd-order differences with
  periodic wrap (``module_big_step_utilities_em.F:1546-1612``).
* ``advective_order in (5, 6)`` with ``specified=True`` (v0.14 real-case fix,
  the WRF ``advective_order <= 6`` branch ``:1768-2072``): map-factored
  (``msfvy``/``msfux`` on the velocity pair, ``1/msfty`` overall) 6th-order
  symmetric interior stencil with the WRF specified-boundary degradation
  (2nd-order one row in; 4th-order two rows in for y; the x two-rows-in
  4th-order blocks are gated ``open_x*`` ONLY in WRF, so specified domains
  have NO x-advection on columns ``ids+2``/``ide-3`` — reproduced exactly),
  plus the top-face row built from ``cfn/cfn1``-extrapolated winds (open-top
  only; under ``top_lid`` the top face stays zero, matching the production
  rigid-lid configuration where ``advance_w`` forces ``rhs(kde)=0``).

v0.14 root-cause context (proofs/v014/switzerland_acoustic_continuation.json):
in terrain-following coordinates the horizontal advection of phi along eta
surfaces and the vertical omega/gw terms are individually ~65x larger than
their sum (h36 Alps: each ~1.5e5, net ~2.3e3 coupled units).  The old
2nd-order/unit-map horizontal operator differed from the WRF real-case one by
~11% of the term — which, after the cancellation, made the NET ph_tend wrong
by ~7.4x its own magnitude and seeded the Switzerland p/ph-first stage
divergence.  The vertical terms and the stage omega are exact (term-level
oracle parity at the bit-identical h36 state).
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


def _horizontal_advection_specified_order6(
    *,
    u: jax.Array,
    v: jax.Array,
    ph_total: jax.Array,
    muu: jax.Array,
    muv: jax.Array,
    c1f: jax.Array,
    c2f: jax.Array,
    rdx: float,
    rdy: float,
    msfux: jax.Array,
    msfvy: jax.Array,
    msfty: jax.Array,
) -> jax.Array:
    """WRF ``rhs_ph`` order<=6 horizontal phi advection, specified BCs.

    Source: ``module_big_step_utilities_em.F:1768-2072``.  Returns the
    horizontal-advection CONTRIBUTION (to be subtracted from ``ph_tend``) on
    interior faces 1..nz-1, shape ``(nz-1, ny, nx)``.  The top-face row is
    handled separately by the caller (open-top only).

    Row/column structure under specified lateral BCs (0-based mass indices):

    * y: 6th-order rows ``3..ny-4``; 4th-order rows ``2`` and ``ny-3``;
      2nd-order rows ``1`` and ``ny-2``; rows ``0``/``ny-1`` untouched.
    * x: 6th-order columns ``3..nx-4``; 2nd-order columns ``1`` and ``nx-2``;
      columns ``0``/``nx-1`` untouched; columns ``2``/``nx-3`` get NO
      x-advection (the WRF 4th-order x blocks are gated ``open_x*`` only).
    """

    nzp1 = int(ph_total.shape[0])
    nz = nzp1 - 1
    ny = int(ph_total.shape[1])
    nx = int(ph_total.shape[2])
    a = ph_total[1:nz]  # faces 1..nz-1, (nz-1, ny, nx)

    c1f_int = c1f[1:nz, None, None]
    c2f_int = c2f[1:nz, None, None]

    # --- y (v) advection -------------------------------------------------
    v_pair = v[1:nz, :, :] + v[0 : nz - 1, :, :]  # (nz-1, ny+1, nx)
    flow_y = (c1f_int * muv[None, :, :] + c2f_int) * v_pair * msfvy[None, :, :]
    flow_n = flow_y[:, 1:, :]  # north face of mass row j -> (nz-1, ny, nx)
    flow_s = flow_y[:, :-1, :]

    pad_y = jnp.pad(a, ((0, 0), (3, 3), (0, 0)))  # zero-pad; padded rows never selected
    d1_y = pad_y[:, 4:-2, :] - pad_y[:, 2:-4, :]  # a[j+1]-a[j-1]
    d2_y = pad_y[:, 5:-1, :] - pad_y[:, 1:-5, :]  # a[j+2]-a[j-2]
    d3_y = pad_y[:, 6:, :] - pad_y[:, :-6, :]  # a[j+3]-a[j-3]
    sten6_y = (45.0 * d1_y - 9.0 * d2_y + d3_y) / 60.0
    sten4_y = (8.0 * d1_y - d2_y) / 12.0
    dn_y = pad_y[:, 4:-2, :] - pad_y[:, 3:-3, :]  # a[j+1]-a[j]
    ds_y = pad_y[:, 3:-3, :] - pad_y[:, 2:-4, :]  # a[j]-a[j-1]

    jj = jnp.arange(ny)[None, :, None]
    six_y = (jj >= 3) & (jj <= ny - 4)
    four_y = (jj == 2) | (jj == ny - 3)
    two_y = (jj == 1) | (jj == ny - 2)
    adv_y = jnp.where(six_y, (flow_n + flow_s) * sten6_y, 0.0)
    adv_y = adv_y + jnp.where(four_y, (flow_n + flow_s) * sten4_y, 0.0)
    adv_y = adv_y + jnp.where(two_y, flow_n * dn_y + flow_s * ds_y, 0.0)

    # --- x (u) advection -------------------------------------------------
    u_pair = u[1:nz, :, :] + u[0 : nz - 1, :, :]  # (nz-1, ny, nx+1)
    flow_x = (c1f_int * muu[None, :, :] + c2f_int) * u_pair * msfux[None, :, :]
    flow_e = flow_x[:, :, 1:]  # east face of mass column i
    flow_w = flow_x[:, :, :-1]

    pad_x = jnp.pad(a, ((0, 0), (0, 0), (3, 3)))
    d1_x = pad_x[:, :, 4:-2] - pad_x[:, :, 2:-4]
    d2_x = pad_x[:, :, 5:-1] - pad_x[:, :, 1:-5]
    d3_x = pad_x[:, :, 6:] - pad_x[:, :, :-6]
    sten6_x = (45.0 * d1_x - 9.0 * d2_x + d3_x) / 60.0
    de_x = pad_x[:, :, 4:-2] - pad_x[:, :, 3:-3]
    dw_x = pad_x[:, :, 3:-3] - pad_x[:, :, 2:-4]

    ii = jnp.arange(nx)[None, None, :]
    six_x = (ii >= 3) & (ii <= nx - 4)
    two_x = (ii == 1) | (ii == nx - 2)
    # WRF quirk: columns 2 and nx-3 get NO x-advection under specified BCs.
    adv_x = jnp.where(six_x, (flow_e + flow_w) * sten6_x, 0.0)
    adv_x = adv_x + jnp.where(two_x, flow_e * de_x + flow_w * dw_x, 0.0)

    msfty_b = msfty[None, :, :]
    return 0.25 * float(rdy) / msfty_b * adv_y + 0.25 * float(rdx) / msfty_b * adv_x


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
    advective_order: int = 2,
    specified: bool = False,
    msfux: jax.Array | None = None,
    msfvy: jax.Array | None = None,
    cfn: float = 0.0,
    cfn1: float = 0.0,
    top_lid: bool = True,
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
        # WRF zeroes ph_tend(kde) (:1527) and the k=2..kte gw loop (kte=kde)
        # then ADDS gw at the top face -- so the open-top face carries gw(kde).
        # Under the production rigid lid advance_w forces rhs(kde)=0, so the
        # legacy zero is kept (and the idealized gates stay byte-identical).
        if bool(top_lid):
            ph_tend = ph_tend.at[nz, :, :].set(0.0)
        else:
            ph_tend = ph_tend.at[nz, :, :].set(gw[nz, :, :])

    # ----- RHS terms 1 & 2: horizontal advection of phi -----
    if int(advective_order) >= 4 and bool(specified):
        # v0.14 real-case branch: WRF order<=6 with map factors and the
        # specified-boundary degradation (see _horizontal_advection_specified_
        # order6).  ``msfux``/``msfvy`` are required here.
        if msfux is None or msfvy is None:
            raise ValueError("rhs_ph_wrf advective_order>=4 requires msfux/msfvy map factors")
        adv = _horizontal_advection_specified_order6(
            u=u,
            v=v,
            ph_total=ph_total,
            muu=muu,
            muv=muv,
            c1f=c1f,
            c2f=c2f,
            rdx=float(rdx),
            rdy=float(rdy),
            msfux=msfux,
            msfvy=msfvy,
            msfty=msfty,
        )
        ph_tend = ph_tend.at[1:nz, :, :].add(-adv)
        if not bool(top_lid) and nz >= 3:
            # Open-top WRF adds (a) the gw term at face kde (handled in the gw
            # block via the same top_lid gate) and (b) the top-face advection
            # row with cfn/cfn1-extrapolated winds and 0.5 weight
            # (module_big_step_utilities_em.F top "k = kte" rows).  Under the
            # production rigid lid this face is forced to zero by advance_w,
            # so it is skipped to keep the lid behaviour unchanged.
            a_top = ph_total[nz]
            v_top = cfn * v[nz - 1] + cfn1 * v[nz - 2]  # (ny+1, nx)
            flow_y = (c1f[nz] * muv + c2f[nz]) * v_top * msfvy
            u_top = cfn * u[nz - 1] + cfn1 * u[nz - 2]  # (ny, nx+1)
            flow_x = (c1f[nz] * muu + c2f[nz]) * u_top * msfux
            pad_y = jnp.pad(a_top, ((3, 3), (0, 0)))
            d1y = pad_y[4:-2, :] - pad_y[2:-4, :]
            d2y = pad_y[5:-1, :] - pad_y[1:-5, :]
            d3y = pad_y[6:, :] - pad_y[:-6, :]
            sten6y = (45.0 * d1y - 9.0 * d2y + d3y) / 60.0
            sten4y = (8.0 * d1y - d2y) / 12.0
            dny = pad_y[4:-2, :] - pad_y[3:-3, :]
            dsy = pad_y[3:-3, :] - pad_y[2:-4, :]
            ny_i = int(a_top.shape[0])
            nx_i = int(a_top.shape[1])
            jj = jnp.arange(ny_i)[:, None]
            adv_y = jnp.where((jj >= 3) & (jj <= ny_i - 4), (flow_y[1:, :] + flow_y[:-1, :]) * sten6y, 0.0)
            adv_y = adv_y + jnp.where((jj == 2) | (jj == ny_i - 3), (flow_y[1:, :] + flow_y[:-1, :]) * sten4y, 0.0)
            adv_y = adv_y + jnp.where(
                (jj == 1) | (jj == ny_i - 2), flow_y[1:, :] * dny + flow_y[:-1, :] * dsy, 0.0
            )
            pad_x = jnp.pad(a_top, ((0, 0), (3, 3)))
            d1x = pad_x[:, 4:-2] - pad_x[:, 2:-4]
            d2x = pad_x[:, 5:-1] - pad_x[:, 1:-5]
            d3x = pad_x[:, 6:] - pad_x[:, :-6]
            sten6x = (45.0 * d1x - 9.0 * d2x + d3x) / 60.0
            dex = pad_x[:, 4:-2] - pad_x[:, 3:-3]
            dwx = pad_x[:, 3:-3] - pad_x[:, 2:-4]
            ii = jnp.arange(nx_i)[None, :]
            adv_x = jnp.where((ii >= 3) & (ii <= nx_i - 4), (flow_x[:, 1:] + flow_x[:, :-1]) * sten6x, 0.0)
            adv_x = adv_x + jnp.where(
                (ii == 1) | (ii == nx_i - 2), flow_x[:, 1:] * dex + flow_x[:, :-1] * dwx, 0.0
            )
            ph_tend = ph_tend.at[nz, :, :].add(
                -(0.5 * float(rdy) / msfty) * adv_y - (0.5 * float(rdx) / msfty) * adv_x
            )
        return ph_tend

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
