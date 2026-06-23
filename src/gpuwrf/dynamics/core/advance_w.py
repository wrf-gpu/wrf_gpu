"""WRF-faithful ``advance_w`` implicit vertical velocity + geopotential solve.

Source: WRF ``dyn_em/module_small_step_em.F:1178-1597`` (subroutine
``advance_w``) plus the large-step vertical PGF/buoyancy source
``pg_buoy_w`` (``module_big_step_utilities_em.F:2498-2578``) and the dry
moisture coefficient ``cqw`` (``calc_cq`` ``:856-902`` then ``pg_buoy_w``
``:2566-2568``).

Array layout (JAX): leading axis is the vertical index.  Faces have shape
``(nz+1, ny, nx)`` and mass levels ``(nz, ny, nx)``.  WRF Fortran index ``k``
runs ``1..kde`` over faces (``kde = nz+1``) and ``1..kde-1`` over mass levels;
the Python translation uses 0-based ``k_face in 0..nz`` and
``k_mass in 0..nz-1``.

For the dry, moisture-free case ``calc_cq`` gives ``cqu=cqv=1`` and ``cqw=0``;
``pg_buoy_w`` then overwrites the interior ``cqw(k)=1/(1+cqw)=1`` for
``k = 2..kde-1``, leaving ``cqw=0`` at the bottom (``k=1``) and top
(``k=kde``) faces.  ``calc_coef_w`` and ``advance_w`` both consume this
post-``pg_buoy_w`` ``cqw``.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from gpuwrf.contracts.precision import force_fp64_island

GRAVITY_M_S2 = 9.81  # WRF ``g`` constant used in the small-step solver.


def _thomas_unroll() -> int:
    """Unroll factor for the two Thomas ``lax.scan`` sweeps (v0.15 perf knob).

    Default ``1`` keeps the v0.14 ``lax.scan(unroll=False)`` lowering.
    Unrolling replicates the loop body in program order without reassociating
    any arithmetic (the same ops on the same values in the same sequence), so
    every factor is bit-identical in fp64 (proofs/perf/v015/ab_s1_flat16.json:
    the only leaf-hash flips are the niter-16 numerics, never the Thomas
    lowering).

    Left at the v0.14 default deliberately for v0.15: the S1 bisect measured
    the unroll as wall-NEUTRAL-to-slightly-worse at the step level (the solve is
    a small fraction; cond16+thomas45=118.6 vs cond16-only=119.8 ms -- within
    noise), while the larger unroll factors materially inflate first-compile
    time (XLA "Very slow compile").  So the unroll buys no step-level wall and is
    not flipped on by default.  Set ``GPUWRF_THOMAS_UNROLL=8`` (or 45) to opt
    into the maximal-fusion Thomas lowering if a specific profile wants it.
    """

    return max(1, int(os.environ.get("GPUWRF_THOMAS_UNROLL", "1")))


def _safe_floors() -> bool:
    """Debug-only guard for the hot-path column-mass / theta-ref divisors.

    The acoustic ``advance_w`` divides by the hybrid-coordinate column masses
    (``mass_h = c1h*muts+c2h``, ``mass_f_mut``, ``mass_f_muts``) and the total
    reference theta (``theta_total_ref = t0 + t_1``).  All four are bounded away
    from zero by construction on valid data: the hybrid column masses satisfy
    ``c1f*mut + c2f >= ptop``-scale and the reference theta is ``300 K`` plus a
    bounded perturbation.  WRF divides by them directly.  The port floored each
    with a per-substep full-grid ``jnp.where(|x|>eps, x, eps)``, which can never
    fire on a valid state but costs four ``abs``+``where`` passes every substep
    (XLA cannot drop them -- the value is data-dependent).

    Default OFF (v0.15): divide directly, matching WRF and bit-identical to the
    floored path WHENEVER no floor would fire (proven on the real case by the
    fp64 A/B leaf-hash gate: any firing floor would inject a NaN/Inf and blow
    the gate, so 0/168 == no floor fired).  Set ``GPUWRF_ADVANCE_W_SAFE_FLOORS=1``
    to restore the unconditional guards for triage of a degenerate state
    (M4 debuggability-hooks convention: zero-cost in production, switchable on).
    """

    return os.environ.get("GPUWRF_ADVANCE_W_SAFE_FLOORS", "0") == "1"


def _floor_pos(x: jax.Array, eps: float) -> jax.Array:
    """Floor ``|x|`` away from ``eps`` only when the debug guard is enabled."""
    if not _safe_floors():
        return x
    return jnp.where(jnp.abs(x) > eps, x, jnp.asarray(eps, dtype=x.dtype))


# WRF vertical-CFL ``w_damping`` constants (``share/module_model_constants.F:88-89``).
W_ALPHA = 0.3   # strength m/s/s
W_BETA = 1.0    # activation CFL number (w_damp_on for non-IEVA)


def w_damp_vertical_cfl(
    rw_tend: jax.Array,
    *,
    ww: jax.Array,
    w: jax.Array,
    mut: jax.Array,
    c1f: jax.Array,
    c2f: jax.Array,
    rdnw: jax.Array,
    dt: float,
    w_alpha: float = W_ALPHA,
    w_crit_cfl: float = W_BETA,
    w_damp_on: float = W_BETA,
) -> jax.Array:
    """Add the WRF ``w_damping=1`` vertical-CFL damping to the large-step ``rw_tend``.

    Source: WRF ``module_big_step_utilities_em.F:2714-2774`` (subroutine
    ``w_damp``).  For interior faces ``k=2..kde-1`` (Python faces ``1..nz-1``):

    ``vert_cfl = |ww/(c1f*mut+c2f) * rdnw * dt|``; when ``vert_cfl > w_damp_on``
    the routine adds ``-sign(w)*w_alpha*(vert_cfl-w_crit_cfl)*(c1f*mut+c2f)`` to
    ``rw_tend``.  ``rw_tend`` and ``w`` are the coupled small-step arrays; ``ww``
    is the coupled small-step omega.  This is a physical CFL limiter, not a clamp:
    it only acts where the vertical Courant number exceeds the activation value.
    """

    nz = int(rw_tend.shape[0]) - 1
    if nz < 2:
        return rw_tend
    # interior faces 1..nz-1 (WRF k=2..kde-1).  WRF accesses c1f/c2f/rdnw at the
    # face index k; rdnw is a mass-level array indexed by the same k value.
    mass_f = c1f[1:nz, None, None] * mut[None, :, :] + c2f[1:nz, None, None]  # (nz-1, ny, nx)
    safe_mass_f = jnp.where(jnp.abs(mass_f) > 1.0e-12, mass_f, jnp.asarray(1.0e-12, dtype=mass_f.dtype))
    ww_int = ww[1:nz, :, :]
    w_int = w[1:nz, :, :]
    rdnw_int = rdnw[1:nz, None, None]
    vert_cfl = jnp.abs(ww_int / safe_mass_f * rdnw_int * float(dt))
    activate = vert_cfl > float(w_damp_on)
    damp = -jnp.sign(w_int) * float(w_alpha) * (vert_cfl - float(w_crit_cfl)) * mass_f
    damp = jnp.where(activate, damp, jnp.zeros_like(damp))
    return rw_tend.at[1:nz, :, :].add(damp)


def dry_cqw(nz: int, ny: int, nx: int, dtype=jnp.float64) -> jax.Array:
    """Return the dry post-``pg_buoy_w`` ``cqw`` face field.

    Interior faces (Python 1..nz-1, WRF k=2..kde-1) are ``1.0``; the bottom
    (k=1) and top (k=kde) faces are ``0.0``.
    """

    cqw = jnp.zeros((nz + 1, ny, nx), dtype=dtype)
    return cqw.at[1:nz, :, :].set(1.0)


def pg_buoy_w_dry(
    p: jax.Array,
    mu_work: jax.Array,
    *,
    c1f: jax.Array,
    rdnw: jax.Array,
    rdn: jax.Array,
    msfty: jax.Array,
    gravity: float = GRAVITY_M_S2,
) -> jax.Array:
    """Return the dry large-step vertical PGF + buoyancy tendency ``rw_tend``.

    Source: WRF ``module_big_step_utilities_em.F:2553-2573`` with ``cqw=0``
    (dry), so ``cq1=1`` and ``cq2=0``:

    * interior face k (2..kde-1):
      ``rw_tend(k) = (1/msfty)*g*( rdn(k)*(p(k)-p(k-1)) - c1f(k)*mu' )``
    * top face k=kde:
      ``rw_tend(kde) = (1/msfty)*g*( 2*rdnw(kde-1)*(-p(kde-1)) - c1f(kde)*mu' )``

    ``rw_tend`` is the *coupled* large-step w tendency (per WRF the solver
    treats ``w`` as the coupled ``mu*w/my`` variable).  ``p`` is the
    perturbation pressure on mass levels ``(nz, ny, nx)``; ``mu_work`` is the
    perturbation dry mass ``(ny, nx)``.
    """

    # v0.20 S2 intrinsic fp64-island lock: the vertical PGF source differences
    # nearly-equal mass-level perturbation pressures (p(k)-p(k-1)). Widen the
    # cancellation inputs to fp64 IN-OPERATOR so an fp32 storage downcast cannot
    # contaminate the buoyancy source. No-op (bit-identical) on fp64_default.
    p, mu_work, c1f, rdnw, rdn, msfty = force_fp64_island(p, mu_work, c1f, rdnw, rdn, msfty)

    nz = int(p.shape[0])
    msft_inv = (1.0 / msfty)[None, :, :]
    rw = jnp.zeros((nz + 1,) + tuple(p.shape[1:]), dtype=p.dtype)
    # interior faces k=1..nz-1 (WRF k=2..kde-1): p(k)-p(k-1) over mass levels.
    interior = msft_inv * float(gravity) * (
        rdn[1:nz, None, None] * (p[1:nz, :, :] - p[: nz - 1, :, :])
        - c1f[1:nz, None, None] * mu_work[None, :, :]
    )
    rw = rw.at[1:nz, :, :].set(interior)
    # top face k=nz (WRF k=kde): uses 2*rdnw(kde-1)*(-p(kde-1)).
    top = msft_inv[:, :, :][0] * float(gravity) * (
        2.0 * rdnw[nz - 1] * (-p[nz - 1, :, :]) - c1f[nz] * mu_work
    )
    rw = rw.at[nz, :, :].set(top)
    return rw


def moist_cqw_calc_face(qtot_mass: jax.Array) -> jax.Array:
    """Return the raw WRF ``calc_cq`` ``cqw`` moist-loading field on w-faces.

    Source: WRF ``calc_cq`` (``module_big_step_utilities_em.F:856-870``) — the
    vertical analogue of the ``cqu``/``cqv`` horizontal moist loading
    (``acoustic_wrf.moisture_coupling_factors``), but on w-faces:

    ``cqw(i,k,j) = 0.5*qtot`` with ``qtot = sum_species( q(k) + q(k-1) )`` for
    interior faces ``k = kts+1..ktf`` (WRF 1-based; Python faces ``1..nz-1``),
    i.e. ``cqw_face(k) = 0.5*(QTOT(k) + QTOT(k-1))`` where ``QTOT`` is total
    moisture ``qv+qc+qr+qi+qs+qg`` on mass levels.

    ``qtot_mass`` is the mass-level total moisture ``(nz, ny, nx)``.  The returned
    ``(nz+1, ny, nx)`` face field is ``0`` at the bottom (``k=0``) and top
    (``k=nz``) faces, which WRF leaves out of the interior loop and which the
    downstream ``calc_coef_w``/``advance_w`` interior solve never consumes.
    """

    nz = int(qtot_mass.shape[0])
    cqw_calc = jnp.zeros((nz + 1,) + tuple(qtot_mass.shape[1:]), dtype=qtot_mass.dtype)
    if nz >= 2:
        cqw_calc = cqw_calc.at[1:nz, :, :].set(
            0.5 * (qtot_mass[1:nz, :, :] + qtot_mass[: nz - 1, :, :])
        )
    return cqw_calc


def pg_buoy_w_moist(
    p: jax.Array,
    mu_work: jax.Array,
    mub: jax.Array,
    cqw_calc: jax.Array,
    *,
    c1f: jax.Array,
    c2f: jax.Array,
    rdnw: jax.Array,
    rdn: jax.Array,
    msfty: jax.Array,
    gravity: float = GRAVITY_M_S2,
) -> tuple[jax.Array, jax.Array]:
    """Return the *moist* large-step vertical PGF + buoyancy ``rw_tend`` and the
    post-``pg_buoy_w`` ``cqw`` field consumed by ``calc_coef_w``/``advance_w``.

    Source: WRF ``pg_buoy_w`` (``module_big_step_utilities_em.F:2474-2497``) full
    moist form with ``cq1 = 1/(1+cqw)`` and ``cq2 = cqw*cq1`` (``cqw`` here is the
    raw ``calc_cq`` field from :func:`moist_cqw_calc_face`):

    * interior face k (Python ``1..nz-1``):
      ``rw_tend(k) = (1/msfty)*g*( cq1*rdn(k)*(p(k)-p(k-1))
                                   - c1f(k)*mu' - cq2*(c1f(k)*mub + c2f(k)) )``
      and the solver ``cqw(k) <- cq1``.
    * top face k=kde (Python ``nz``), using ``cqw(kde-1)``:
      ``rw_tend(kde) = (1/msfty)*g*( cq1*2*rdnw(kde-1)*(-p(kde-1))
                                     - c1f(kde)*mu' - cq2*(c1f(kde)*mub + c2f(kde)) )``

    The extra ``-cq2*(c1f*mub + c2f)`` term is the **water-mass loading** the dry
    specialization (:func:`pg_buoy_w_dry`, ``cqw=0`` -> ``cq1=1, cq2=0``) omits;
    without it the acoustic solver relaxes the column to *dry* hydrostatic
    balance rather than WRF's *moist* balance.  With ``cqw_calc == 0`` this
    function is bit-identical to ``pg_buoy_w_dry`` + ``dry_cqw``.

    ``p`` is the perturbation pressure on mass levels ``(nz, ny, nx)``;
    ``mu_work`` is the perturbation dry mass ``mu'`` ``(ny, nx)``; ``mub`` is the
    base-state dry mass ``(ny, nx)``; ``cqw_calc`` is the ``(nz+1, ny, nx)`` raw
    moist-loading face field.
    """

    nz = int(p.shape[0])
    msft_inv = (1.0 / msfty)[None, :, :]
    g = float(gravity)
    rw = jnp.zeros((nz + 1,) + tuple(p.shape[1:]), dtype=p.dtype)
    cqw_solver = jnp.zeros((nz + 1,) + tuple(p.shape[1:]), dtype=p.dtype)

    if nz >= 2:
        cqw_int = cqw_calc[1:nz, :, :]
        cq1_int = 1.0 / (1.0 + cqw_int)
        cq2_int = cqw_int * cq1_int
        interior = msft_inv * g * (
            cq1_int * rdn[1:nz, None, None] * (p[1:nz, :, :] - p[: nz - 1, :, :])
            - c1f[1:nz, None, None] * mu_work[None, :, :]
            - cq2_int * (c1f[1:nz, None, None] * mub[None, :, :] + c2f[1:nz, None, None])
        )
        rw = rw.at[1:nz, :, :].set(interior)
        cqw_solver = cqw_solver.at[1:nz, :, :].set(cq1_int)

    # top face k=nz (WRF k=kde) uses the raw cqw at the topmost interior face nz-1.
    cqw_top = cqw_calc[nz - 1, :, :]
    cq1_top = 1.0 / (1.0 + cqw_top)
    cq2_top = cqw_top * cq1_top
    top = msft_inv[0] * g * (
        cq1_top * 2.0 * rdnw[nz - 1] * (-p[nz - 1, :, :])
        - c1f[nz] * mu_work
        - cq2_top * (c1f[nz] * mub + c2f[nz])
    )
    rw = rw.at[nz, :, :].set(top)
    return rw, cqw_solver


def advance_w_wrf(
    *,
    w: jax.Array,
    rw_tend: jax.Array,
    ww: jax.Array,
    u: jax.Array,
    v: jax.Array,
    mu_work: jax.Array,
    mut: jax.Array,
    muave: jax.Array,
    muts: jax.Array,
    t_2ave: jax.Array,
    t_2: jax.Array,
    t_1: jax.Array,
    ph: jax.Array,
    ph_1: jax.Array,
    phb: jax.Array,
    ph_tend: jax.Array,
    ht: jax.Array,
    c2a: jax.Array,
    cqw: jax.Array,
    alt: jax.Array,
    a: jax.Array,
    alpha: jax.Array,
    gamma: jax.Array,
    c1h: jax.Array,
    c2h: jax.Array,
    c1f: jax.Array,
    c2f: jax.Array,
    rdnw: jax.Array,
    rdn: jax.Array,
    fnm: jax.Array,
    fnp: jax.Array,
    cf1: jax.Array,
    cf2: jax.Array,
    cf3: jax.Array,
    msftx: jax.Array,
    msfty: jax.Array,
    rdx: float,
    rdy: float,
    dts: float,
    epssm: float,
    t0: float = 300.0,
    top_lid: bool = False,
    gravity: float = GRAVITY_M_S2,
    w_save: jax.Array | None = None,
    damp_opt: int = 0,
    dampcoef: float = 0.0,
    zdamp: float = 5000.0,
    w_damping: int = 0,
    w_alpha: float = W_ALPHA,
    w_crit_cfl: float = W_BETA,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Advance implicit ``w`` and geopotential ``ph`` for one acoustic substep.

    Returns ``(w_next, ph_next, t_2ave_next)``.  ``w`` and ``ph`` are the
    *coupled* small-step work arrays; the caller decouples them in
    ``small_step_finish``.  Terrain (``ht``) is honoured at the lower
    boundary; the top is rigid only when ``top_lid``.

    Damping (Block 1, WRF-faithful, citations inline):
    * ``w_damping=1``: vertical-CFL limiter added to ``rw_tend`` before the solve
      (``module_big_step_utilities_em.F:2714-2774``).
    * ``damp_opt=3``: implicit Rayleigh ``w`` damping at the model top applied
      after the Thomas back-substitution, before the geopotential finish
      (``module_small_step_em.F:1559-1572``), using ``dampmag=dts*dampcoef`` and
      ``hdepth=zdamp``.  ``w_save`` is the uncoupled physical perturbation ``w``
      from ``small_step_prep`` (``module_small_step_em.F:272``).

    WRF source lines are cited inline.
    """

    # v0.20 S2 intrinsic fp64-island lock: the implicit w/phi solve is a chain of
    # nearly-cancelling vertical differences -- the geopotential predictor RHS, the
    # buoyancy term-B (rdn*(c2a*alt*t2ave)(k) - (...)(k-1)), the dphi geopotential
    # gradient, and the Thomas tridiagonal (a/alpha/gamma). Widen every
    # cancellation-sensitive solve input to fp64 IN-OPERATOR so a later fp32
    # storage downcast cannot contaminate the genuine-fp64 vertical solve. No-op
    # (bit-identical, no convert ops) on fp64_default: all inputs already fp64.
    (
        w, rw_tend, ww, mu_work, mut, muave, muts, t_2ave, t_2, t_1,
        ph, ph_1, phb, ph_tend, c2a, alt, a, alpha, gamma,
        c1h, c2h, c1f, c2f, rdnw, rdn, fnm, fnp, cf1, cf2, cf3, msfty, w_save,
    ) = force_fp64_island(
        w, rw_tend, ww, mu_work, mut, muave, muts, t_2ave, t_2, t_1,
        ph, ph_1, phb, ph_tend, c2a, alt, a, alpha, gamma,
        c1h, c2h, c1f, c2f, rdnw, rdn, fnm, fnp, cf1, cf2, cf3, msfty, w_save,
    )

    nz = int(w.shape[0]) - 1  # number of mass levels; faces 0..nz.
    g = float(gravity)
    msft_inv = (1.0 / msfty)[None, :, :]
    eps_p = 1.0 + float(epssm)
    eps_m = 1.0 - float(epssm)

    # --- WRF w_damping=1 vertical-CFL damping of the large-step rw_tend ---
    # (module_big_step_utilities_em.F:2766-2770).  Acts on rw_tend before the
    # implicit solve; only where the vertical Courant number exceeds w_beta.
    if int(w_damping) == 1:
        rw_tend = w_damp_vertical_cfl(
            rw_tend,
            ww=ww,
            w=w,
            mut=mut,
            c1f=c1f,
            c2f=c2f,
            rdnw=rdnw,
            dt=float(dts),
            w_alpha=float(w_alpha),
            w_crit_cfl=float(w_crit_cfl),
            w_damp_on=float(W_BETA),
        )

    # --- t_2ave (WRF :1341-1344) on mass levels k=1..k_end (Python 0..nz-1) ---
    mass_h = c1h[:, None, None] * muts[None, :, :] + c2h[:, None, None]
    safe_mass_h = _floor_pos(mass_h, 1.0e-12)
    t_2ave_half = 0.5 * (eps_p * t_2 + eps_m * t_2ave)
    theta_total_ref = float(t0) + t_1
    safe_theta_ref = _floor_pos(theta_total_ref, 1.0e-6)
    t_2ave_next = (t_2ave_half + (c1h[:, None, None] * muave[None, :, :]) * float(t0)) / (
        safe_mass_h * safe_theta_ref
    )

    # --- RHS of phi equation (WRF :1345) on faces 1..nz ---
    # rhs[f] = dts*(ph_tend[f] + 0.5*g*(1-epssm)*w[f]) for f=1..nz
    rhs = jnp.zeros_like(w)
    rhs_main = float(dts) * (ph_tend[1:, :, :] + 0.5 * g * eps_m * w[1:, :, :])
    rhs = rhs.at[1:, :, :].set(rhs_main)
    # rhs(i,1) = 0 (WRF :1333)
    rhs = rhs.at[0, :, :].set(0.0)

    # NOTE (v0.15 kernel probe, NEGATIVE result, kept for the record): hoisting
    # the stage-constant denominators (mass_f_mut / safe_mass_f_mut /
    # safe_mass_h_mut / coef_mass) out of the substep scan to once-per-stage
    # was tried and REVERTED: it measured SLOWER (178.9 -> 188.7 ms/step,
    # proofs/perf/v015/ab_streamA.json) because the inline broadcast-FMA
    # recompute fuses into the consumers for free while the hoisted arrays
    # cost real DRAM loads -- and it was NOT bit-identical in-program
    # (proofs/perf/v015/ab_compare_v014_base_vs_streamA.json: XLA FMA
    # contraction differs across fusion contexts). Keep these computed
    # in place, exactly as WRF/v0.14.

    # --- phi advection term (WRF ELSE branch, phi_adv_z==1, :1370-1382) ---
    # wdwn(k+1) = 0.5*(ww(k+1)+ww(k))*rdnw(k)*(ph_1(k+1)-ph_1(k)+phb(k+1)-phb(k))
    # for k=1..k_end (Python mass index 0..nz-1) -> stored at face k+1 (1..nz).
    ph_total_1 = ph_1 + phb
    dphi = ph_total_1[1:, :, :] - ph_total_1[:-1, :, :]  # (nz, ny, nx)
    ww_mid = 0.5 * (ww[1:, :, :] + ww[:-1, :, :])  # (nz, ny, nx) face-avg per mass level
    wdwn = jnp.zeros_like(w)
    wdwn = wdwn.at[1:, :, :].set(ww_mid * rdnw[:, None, None] * dphi)
    # rhs(k) -= dts*(fnm(k)*wdwn(k+1)+fnp(k)*wdwn(k)) for k=2..k_end (faces 1..nz-1)
    if nz >= 2:
        rhs_adv = float(dts) * (
            fnm[1:nz, None, None] * wdwn[2 : nz + 1, :, :] + fnp[1:nz, None, None] * wdwn[1:nz, :, :]
        )
        rhs = rhs.at[1:nz, :, :].add(-rhs_adv)

    # --- finalize rhs as the explicit ph predictor (WRF :1393-1398) ---
    # rhs(k) = ph(k) + msfty*rhs(k)/(c1f(k)*mut+c2f(k)) for k=2..k_end+1 (faces 1..nz)
    mass_f_mut = c1f[:, None, None] * mut[None, :, :] + c2f[:, None, None]  # (nz+1, ny, nx)
    safe_mass_f_mut = _floor_pos(mass_f_mut, 1.0e-12)
    rhs = rhs.at[1:, :, :].set(
        ph[1:, :, :] + msfty[None, :, :] * rhs[1:, :, :] / safe_mass_f_mut[1:, :, :]
    )
    if bool(top_lid):
        rhs = rhs.at[nz, :, :].set(0.0)

    # --- lower boundary condition on w from terrain (WRF :1417-1429) ---
    # w(i,1,j) = msfty*0.5*rdy*( (ht(j+1)-ht(j))*(cf1*v(1,j+1)+cf2*v(2,j+1)+cf3*v(3,j+1))
    #                          +(ht(j)-ht(j-1))*(cf1*v(1,j)+cf2*v(2,j)+cf3*v(3,j)) )
    #          + msftx*0.5*rdx*( (ht(i+1)-ht(i))*(cf1*u(i+1,1)+cf2*u(i+1,2)+cf3*u(i+1,3))
    #                          +(ht(i)-ht(i-1))*(cf1*u(i,1)+cf2*u(i,2)+cf3*u(i,3)) )
    # v is staggered on y (ny+1, nx); u is staggered on x (ny, nx+1). For flat
    # terrain (ht uniform) every difference is zero -> w_surface = 0, which is
    # the periodic/idealized gate path.  Honour the general terrain form.
    ht_dy_n = jnp.pad(ht, ((0, 1), (0, 0)), mode="edge")[1:, :] - ht  # ht(j+1)-ht(j) at mass (ny,nx)
    ht_dy_s = ht - jnp.pad(ht, ((1, 0), (0, 0)), mode="edge")[:-1, :]  # ht(j)-ht(j-1)
    ht_dx_e = jnp.pad(ht, ((0, 0), (0, 1)), mode="edge")[:, 1:] - ht  # ht(i+1)-ht(i)
    ht_dx_w = ht - jnp.pad(ht, ((0, 0), (1, 0)), mode="edge")[:, :-1]  # ht(i)-ht(i-1)

    def _cf_combo_3(field_lo3: jax.Array) -> jax.Array:
        # cf1*level0 + cf2*level1 + cf3*level2 of a (>=3, ...) array
        return cf1 * field_lo3[0] + cf2 * field_lo3[1] + cf3 * field_lo3[2]

    # v at south/north faces nearest surface (use v rows j and j+1 for each mass cell).
    v_south = v[:, :-1, :]  # (nz_or_nz+? , ny, nx) v has mass-vertical levels
    v_north = v[:, 1:, :]
    v_cf_n = _cf_combo_3(v_north[:3, :, :])  # (ny, nx)
    v_cf_s = _cf_combo_3(v_south[:3, :, :])
    u_west = u[:, :, :-1]
    u_east = u[:, :, 1:]
    u_cf_e = _cf_combo_3(u_east[:3, :, :])
    u_cf_w = _cf_combo_3(u_west[:3, :, :])
    w_surface = (
        msfty * 0.5 * float(rdy) * (ht_dy_n * v_cf_n + ht_dy_s * v_cf_s)
        + msftx * 0.5 * float(rdx) * (ht_dx_e * u_cf_e + ht_dx_w * u_cf_w)
    )

    # --- explicit w update on interior faces k=2..k_end (Python faces 1..nz-1) (WRF :1477-1489) ---
    # term A (implicit pressure via c2a): msft_inv*cqw*0.5*dts*g*rdn(k)*( c2a(k)*rdnw(k)/(c1h(k)*mut+c2h(k))
    #     *((1+eps)*(rhs(k+1)-rhs(k))+(1-eps)*(ph(k+1)-ph(k)))
    #   - c2a(k-1)*rdnw(k-1)/(c1h(k-1)*mut+c2h(k-1))
    #     *((1+eps)*(rhs(k)-rhs(k-1))+(1-eps)*(ph(k)-ph(k-1))) )
    # term B (buoyancy): dts*g*msft_inv*( rdn(k)*(c2a(k)*alt(k)*t2ave(k)-c2a(k-1)*alt(k-1)*t2ave(k-1)) - c1f(k)*muave )
    mass_h_mut = c1h[:, None, None] * mut[None, :, :] + c2h[:, None, None]  # (nz, ny, nx)
    safe_mass_h_mut = jnp.where(
        jnp.abs(mass_h_mut) > 1.0e-12, mass_h_mut, jnp.asarray(1.0e-12, dtype=mass_h_mut.dtype)
    )
    # coefficient per mass level k: c2a(k)*rdnw(k)/(c1h(k)*mut+c2h(k))
    coef_mass = c2a * rdnw[:, None, None] / safe_mass_h_mut  # (nz, ny, nx) indexed by mass level

    w_next = w + float(dts) * rw_tend

    if nz >= 2:
        # interior faces f = 1..nz-1 correspond to WRF k=2..k_end, with k mass index = f (upper) and f-1 (lower)
        # upper mass level index = f (0-based mass index f), lower = f-1
        upper = slice(1, nz)  # mass index for "k" -> f
        lower = slice(0, nz - 1)  # mass index for "k-1" -> f-1
        rhs_kp1 = rhs[2 : nz + 1, :, :]  # rhs(k+1) faces 2..nz
        rhs_k = rhs[1:nz, :, :]  # rhs(k) faces 1..nz-1
        rhs_km1 = rhs[0 : nz - 1, :, :]  # rhs(k-1) faces 0..nz-2
        ph_kp1 = ph[2 : nz + 1, :, :]
        ph_k = ph[1:nz, :, :]
        ph_km1 = ph[0 : nz - 1, :, :]

        termA_upper = coef_mass[upper, :, :] * (eps_p * (rhs_kp1 - rhs_k) + eps_m * (ph_kp1 - ph_k))
        termA_lower = coef_mass[lower, :, :] * (eps_p * (rhs_k - rhs_km1) + eps_m * (ph_k - ph_km1))
        termA = msft_inv * cqw[1:nz, :, :] * (0.5 * float(dts) * g * rdn[1:nz, None, None]) * (
            termA_upper - termA_lower
        )

        buoy_upper = c2a[upper, :, :] * alt[upper, :, :] * t_2ave_next[upper, :, :]
        buoy_lower = c2a[lower, :, :] * alt[lower, :, :] * t_2ave_next[lower, :, :]
        termB = float(dts) * g * msft_inv * (
            rdn[1:nz, None, None] * (buoy_upper - buoy_lower) - (c1f[1:nz, None, None] * muave[None, :, :])
        )
        w_next = w_next.at[1:nz, :, :].add(termA + termB)

    # --- top face k=kde (Python face nz) (WRF :1492-1502) ---
    # w(kde) += dts*rw_tend(kde) + msft_inv*(
    #    -0.5*dts*g/(c1h(kde-1)*mut+c2h(kde-1))*rdnw(kde-1)^2*2*c2a(kde-1)
    #        *((1+eps)*(rhs(kde)-rhs(kde-1))+(1-eps)*(ph(kde)-ph(kde-1)))
    #    -dts*g*(2*rdnw(kde-1)*c2a(kde-1)*alt(kde-1)*t2ave(kde-1) + c1f(kde)*muave) )
    km1 = nz - 1  # mass index kde-1
    rhs_top = rhs[nz, :, :]
    rhs_topm1 = rhs[nz - 1, :, :]
    ph_top = ph[nz, :, :]
    ph_topm1 = ph[nz - 1, :, :]
    termA_top = (
        -0.5 * float(dts) * g / safe_mass_h_mut[km1, :, :]
        * rdnw[km1] ** 2 * 2.0 * c2a[km1, :, :]
        * (eps_p * (rhs_top - rhs_topm1) + eps_m * (ph_top - ph_topm1))
    )
    termB_top = -float(dts) * g * (
        2.0 * rdnw[km1] * c2a[km1, :, :] * alt[km1, :, :] * t_2ave_next[km1, :, :]
        + (c1f[nz] * muave)
    )
    w_top = w[nz, :, :] + float(dts) * rw_tend[nz, :, :] + msft_inv[0] * (termA_top + termB_top)
    if bool(top_lid):
        w_top = jnp.zeros_like(w_top)
    w_next = w_next.at[nz, :, :].set(w_top)

    # surface w (WRF :1417-1429) at face 0.
    w_next = w_next.at[0, :, :].set(w_surface)

    # --- Thomas forward sweep (WRF :1533-1537): w(k)=(w(k)-a(k)*w(k-1))*alpha(k) for k=2..kde ---
    def _fwd(prev_w, entries):
        a_k, alpha_k, w_k = entries
        out = (w_k - a_k * prev_w) * alpha_k
        return out, out

    _u = _thomas_unroll()
    _scan_unroll = _u if _u > 1 else False  # 1 -> False == the exact v0.14 lowering
    _, fwd_tail = jax.lax.scan(
        _fwd, w_next[0], (a[1:, :, :], alpha[1:, :, :], w_next[1:, :, :]), unroll=_scan_unroll
    )
    w_fwd = jnp.concatenate((w_next[0][None, ...], fwd_tail), axis=0)

    # --- Thomas back substitution (WRF :1546-1550): w(k)=w(k)-gamma(k)*w(k+1) for k=k_end..2 ---
    # i.e. faces nz-1 down to 1. Faces 0 and nz are fixed boundaries.
    # NOTE (v0.15 kernel probe): a ``reverse=True`` rewrite of this flip-scan-
    # flip removes three full-grid reversed copies per substep and is bitwise-
    # equal at the lax level (tests/test_v015_stream_a_bitwise.py), but in the
    # FULL program the changed fusion context shifted last bits (XLA FMA
    # contraction is fusion-context-dependent) and it measured wall-neutral at
    # 128^2 -- so the v0.14 formulation is kept verbatim; revisit inside the
    # planned Pallas column solve where the op order is forced explicitly.
    def _back(next_w, entries):
        gamma_k, w_k = entries
        out = w_k - gamma_k * next_w
        return out, out

    if nz >= 2:
        gamma_rev = gamma[1:nz, :, :][::-1]
        w_rev = w_fwd[1:nz, :, :][::-1]
        _, interior_rev = jax.lax.scan(_back, w_fwd[nz], (gamma_rev, w_rev), unroll=_scan_unroll)
        interior = interior_rev[::-1]
        w_solved = jnp.concatenate((w_fwd[0][None, ...], interior, w_fwd[nz][None, ...]), axis=0)
    else:
        w_solved = w_fwd

    # --- damp_opt=3 implicit Rayleigh w-damping at the model top (WRF :1559-1572) ---
    # Applied after the Thomas back-substitution, before the geopotential finish.
    #   dampmag = dts*dampcoef; hdepth = zdamp
    #   htop = (ph_1(kde)+phb(kde))/g; hk = (ph_1(k)+phb(k))/g; hbot = htop-hdepth
    #   dampwt(k) = dampmag*sin(0.5*pi*(hk-hbot)/hdepth)^2 for hk>=hbot else 0
    #   w(k) = (w(k) - dampwt(k)*(c1f(k)*mut(i,j)+c2f(k))*w_save(k)) / (1+dampwt(k))
    # w_save is the uncoupled physical perturbation w from small_step_prep (:272);
    # (c1f*mut+c2f) couples it to the same coupled representation as the solved w.
    if int(damp_opt) == 3 and w_save is not None and float(dampcoef) > 0.0:
        dampmag = float(dts) * float(dampcoef)
        hdepth = float(zdamp)
        ph_total_damp = ph_1 + phb  # (nz+1, ny, nx)
        htop = ph_total_damp[nz, :, :] / g  # (ny, nx)
        hk = ph_total_damp / g  # (nz+1, ny, nx)
        hbot = (htop - hdepth)[None, :, :]
        pi = jnp.asarray(jnp.pi, dtype=w_solved.dtype)
        ramp = jnp.sin(0.5 * pi * (hk - hbot) / hdepth) ** 2
        dampwt = jnp.where(hk >= hbot, dampmag * ramp, jnp.zeros_like(ramp))
        mass_f_mut_damp = c1f[:, None, None] * mut[None, :, :] + c2f[:, None, None]
        w_damped = (w_solved - dampwt * mass_f_mut_damp * w_save) / (1.0 + dampwt)
        # WRF damps faces k=2..kde+? loop is k=kde+1,2,-1 -> all faces 1..nz; the
        # surface face k=1 (Python 0) is the terrain BC and is reset below, so
        # keep faces 1..nz here.
        w_solved = w_solved.at[1:, :, :].set(w_damped[1:, :, :])

    # --- geopotential finish (WRF :1581-1586): ph(k)=rhs(k)+msfty*0.5*dts*g*(1+eps)*w(k)/(c1f(k)*muts+c2f(k))
    # for k=k_end+1..2 (faces 1..nz). ---
    mass_f_muts = c1f[:, None, None] * muts[None, :, :] + c2f[:, None, None]
    safe_mass_f_muts = _floor_pos(mass_f_muts, 1.0e-12)
    ph_next = ph
    ph_upd = rhs[1:, :, :] + msfty[None, :, :] * 0.5 * float(dts) * g * eps_p * w_solved[1:, :, :] / safe_mass_f_muts[1:, :, :]
    ph_next = ph_next.at[1:, :, :].set(ph_upd)

    return w_solved, ph_next, t_2ave_next


__all__ = [
    "advance_w_wrf",
    "pg_buoy_w_dry",
    "pg_buoy_w_moist",
    "moist_cqw_calc_face",
    "dry_cqw",
    "GRAVITY_M_S2",
]
